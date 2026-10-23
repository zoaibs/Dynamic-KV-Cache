"""Real inference smoke test with upstream native, unquantized cache."""
import json
import time
from pathlib import Path
import torch
from src.model_adapter import VideoModel, decode_video, native_cache_bytes
from scripts.resources import ROOT, sha256

def main():
    config = json.loads((ROOT / 'configs/default.json').read_text())
    sample = json.loads((ROOT / 'data/manifest.json').read_text())['samples'][0]
    t = time.perf_counter()
    model = VideoModel(config)
    load_seconds = time.perf_counter() - t
    frames = decode_video(sample['path'], config['frames'])
    cache = model.begin()
    trace = []
    for frame in frames:
        old_key = cache.key_cache[0].clone()
        n = cache.get_seq_length()
        trace.append(model.append(frame, cache))
        assert torch.equal(old_key, cache.key_cache[0][..., :n, :])
    visual_calls = model.vision_calls
    cache_bytes = native_cache_bytes(cache)
    result = model.answer(sample, cache)
    assert model.vision_calls == visual_calls
    # Separate fresh native stream for a free-text sanity prediction.
    descriptive_cache = model.begin()
    for frame in frames:
        model.append(frame, descriptive_cache)
    description = model.describe(descriptive_cache)
    record = dict(status='success', config=config, sample_id=sample['id'],
                  load_seconds=load_seconds, elapsed_seconds=time.perf_counter()-t,
                  cache_bytes_before_question=cache_bytes,
                  native_cache_type=type(cache).__name__, frame_trace=trace,
                  historical_keys_unchanged=True, qa_reuses_visual_cache=True,
                  vision_calls_for_video=visual_calls, description=description,
                  answer=result, checkpoint_revision=config['revision'],
                  adapter_sha256=sha256(ROOT / 'src/model_adapter.py'))
    (ROOT / 'results/raw/baseline_smoke.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)

if __name__ == '__main__':
    main()
