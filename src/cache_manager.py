"""Mixed-frame cache in the real Transformers Cache.update attention path."""
from dataclasses import dataclass
import struct
import time
import torch
from transformers.cache_utils import DynamicCache
from src.kv_quantization import pack, unpack

SEGMENT = struct.Struct('<iiqqB7x')  # layer, frame, absolute start/end, bits: 32 B

@dataclass
class Segment:
    key: object
    value: object
    descriptor: torch.Tensor

class MixedPrecisionCache(DynamicCache):
    def __init__(self, group_size=64):
        super().__init__()
        self.segments = {}
        self.group_size = group_size
        self.pending_bits = 16
        self.frame_id = -1
        self.pack_seconds = 0.0
        self.unpack_seconds = 0.0
        self.retrievals = 0

    def select(self, bits, frame_id=-1):
        if bits not in (4, 8, 16):
            raise ValueError('Invalid precision')
        self.pending_bits, self.frame_id = bits, frame_id

    def get_seq_length(self, layer_idx=0):
        segments = self.segments.get(layer_idx, [])
        return 0 if not segments else SEGMENT.unpack(bytes(segments[-1].descriptor.tolist()))[3]

    def get_max_cache_shape(self):
        return None

    def __len__(self):
        return len(self.segments)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        historical = self.segments.setdefault(layer_idx, [])
        start = self.get_seq_length(layer_idx)
        # Dequantize previous frames only. The newly computed frame remains native
        # for its own attention; its compressed representation is used next call.
        t = time.perf_counter()
        keys = [unpack(s.key, key_states.dtype, key_states.device) for s in historical]
        values = [unpack(s.value, value_states.dtype, value_states.device) for s in historical]
        self.unpack_seconds += time.perf_counter() - t
        self.retrievals += len(historical)
        t = time.perf_counter()
        k = pack(key_states, self.pending_bits, self.group_size)
        v = pack(value_states, self.pending_bits, self.group_size)
        end = start + key_states.shape[-2]
        raw = SEGMENT.pack(layer_idx, self.frame_id, start, end, self.pending_bits)
        historical.append(Segment(k, v, torch.tensor(list(raw), dtype=torch.uint8)))
        self.pack_seconds += time.perf_counter() - t
        if layer_idx == 0:
            self._seen_tokens = end
        # Each reconstructed history lives only for the current layer's attention.
        return torch.cat([*keys, key_states], dim=-2), torch.cat([*values, value_states], dim=-2)

    def __getitem__(self, layer_idx):
        if layer_idx not in self.segments:
            raise KeyError(layer_idx)
        segments = self.segments[layer_idx]
        return (torch.cat([unpack(s.key) for s in segments], dim=-2),
                torch.cat([unpack(s.value) for s in segments], dim=-2))

    def __iter__(self):
        for layer in sorted(self.segments):
            yield self[layer]

    def accounting(self, frame_only=False):
        payload = metadata = 0
        by_bits = {4: 0, 8: 0, 16: 0}
        for segments in self.segments.values():
            for s in segments:
                _, frame, _, _, bits = SEGMENT.unpack(bytes(s.descriptor.tolist()))
                if frame_only and frame < 0:
                    continue
                size = s.key.storage_bytes + s.value.storage_bytes + s.descriptor.numel()
                payload += s.key.payload_bytes + s.value.payload_bytes
                metadata += s.key.metadata_bytes + s.value.metadata_bytes + s.descriptor.numel()
                by_bits[bits] += size
        return dict(cpu_payload_bytes=payload, cpu_metadata_bytes=metadata,
                    cpu_cache_bytes=payload+metadata, gpu_cache_bytes=0, disk_cache_bytes=0,
                    bytes_by_precision=by_bits)
