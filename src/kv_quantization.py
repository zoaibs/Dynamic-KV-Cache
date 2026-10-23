"""Groupwise affine quantization with real nibble packing and explicit headers.

Only CPU storage is implemented. Attention receives reconstructed native dtype
tensors. This is a storage prototype, without a fused low-bit attention kernel.
"""
from dataclasses import dataclass
import math
import struct
import torch
import torch.nn.functional as F

HEADER = struct.Struct('<4sBBBB4qQQQ')  # 64 bytes, including shape and dtype
DTYPES = {torch.float16: 1, torch.bfloat16: 2, torch.float32: 3}

@dataclass
class PackedTensor:
    payload: torch.Tensor
    scales: torch.Tensor
    offsets: torch.Tensor
    header: torch.Tensor

    def fields(self):
        magic, version, bits, dtype, ndim, *other = HEADER.unpack(bytes(self.header.tolist()))
        if magic != b'DKVC' or version != 1:
            raise ValueError('Invalid packed tensor header')
        shape, group, numel, groups = tuple(other[:ndim]), other[4], other[5], other[6]
        return bits, dtype, shape, group, numel, groups

    @property
    def payload_bytes(self):
        return self.payload.numel() * self.payload.element_size()

    @property
    def metadata_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.scales, self.offsets, self.header))

    @property
    def storage_bytes(self):
        return self.payload_bytes + self.metadata_bytes

def pack(tensor, bits, group_size=64):
    if bits not in (4, 8, 16) or group_size <= 0:
        raise ValueError('Supported bit widths: 4, 8, 16; positive group size required')
    if tensor.ndim != 4 or not tensor.is_floating_point() or tensor.numel() == 0:
        raise ValueError('Expected nonempty [batch, heads, tokens, head_dim] float tensor')
    x = tensor.detach().to('cpu').contiguous()
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite KV tensor')
    if bits == 16:
        # Preserve BF16 exactly in this project's native BF16 baseline.
        if x.dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.float16)
        payload = x.view(torch.uint8).flatten().clone()
        scales = offsets = torch.empty(0, dtype=torch.float32)
        groups = 0
    else:
        flat = x.float().flatten()
        groups = math.ceil(flat.numel() / group_size)
        # Replicate the last real value, so padding cannot change range estimates.
        if flat.numel() % group_size:
            flat = torch.cat((flat, flat[-1:].repeat(groups * group_size - flat.numel())))
        grouped = flat.reshape(groups, group_size)
        offsets = grouped.amin(dim=1)
        spans = grouped.amax(dim=1) - offsets
        scales = torch.where(spans > 0, spans / ((1 << bits) - 1), torch.ones_like(spans))
        q = ((grouped - offsets[:, None]) / scales[:, None]).round().clamp(0, (1 << bits)-1)
        q = q.to(torch.uint8).flatten()[:x.numel()]
        if bits == 4:
            if q.numel() % 2:
                q = F.pad(q, (0, 1))
            payload = (q[0::2] | (q[1::2] << 4)).contiguous()
        else:
            payload = q.contiguous()
    raw = HEADER.pack(b'DKVC', 1, bits, DTYPES[x.dtype], x.ndim,
                      *x.shape, group_size, x.numel(), groups)
    header = torch.tensor(list(raw), dtype=torch.uint8)
    return PackedTensor(payload, scales, offsets, header)

def unpack(packed, dtype=None, device='cpu'):
    bits, original_dtype, shape, group_size, numel, groups = packed.fields()
    source_dtype = next(k for k, v in DTYPES.items() if v == original_dtype)
    dtype = source_dtype if dtype is None else dtype
    if bits == 16:
        return packed.payload.view(source_dtype).reshape(shape).to(dtype=dtype, device=device)
    if bits == 4:
        q = torch.stack((packed.payload & 15, packed.payload >> 4), dim=-1).flatten()[:numel]
    else:
        q = packed.payload
    pad = groups * group_size - numel
    q = F.pad(q, (0, pad)).float().reshape(groups, group_size)
    x = q * packed.scales[:, None] + packed.offsets[:, None]
    return x.flatten()[:numel].reshape(shape).to(dtype=dtype, device=device)
