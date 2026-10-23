# Dynamic KV Cache Scheduler

A real pretrained video-QA prototype with persistent per-frame, per-layer KV
storage at 4, 8 and 16 bits. Four-bit entries actually pack two values per byte.
The scheduler uses exactly Canny edges, Farneback flow and grayscale difference.
No model training, paid API, external judge or cloud compute is involved.

**Start with [FINAL_SUMMARY.md](FINAL_SUMMARY.md)** for measured outcomes and
limitations. Machine-readable measurements live in [results/metrics.csv](results/metrics.csv),
with all 96 timed requests under `results/raw/benchmark.jsonl`. The research paper
is in `paper/`. Resume performance numbers in the original mission were targets,
not benchmark results, and are not adopted as claims here.

## Local setup

This workspace includes an isolated portable Python 3.11.9 under `.runtime/python`
and public checkpoint/data downloads under `models/` and `data/`. They are excluded
from version control. No system-wide Python, Git, PATH, security or power setting
is changed. Use PowerShell in this project's existing VS Code terminal:

```powershell
.\.runtime\python\python.exe -m pip check
.\.runtime\python\python.exe -m scripts.environment
```

For a fresh checkout, inspect then run `scripts/setup.ps1` from PowerShell. It
downloads official portable CPython, verifies its pinned archive hash, initializes
project-local pip and installs `requirements.txt` without a package cache. The
script refuses a changed pip bootstrap hash so an updated public script can be
reviewed before execution. It does not change execution policy; if your policy
blocks scripts, run the reviewed commands directly in the terminal instead.
Dependencies are pinned in `requirements.txt`; the legacy `docopt` source build
is required by `num2words` and uses only local build tools and temp space.

Acquisition commands, if the runtime already exists:

```powershell
.\.runtime\python\python.exe -m scripts.resources
.\.runtime\python\python.exe -m scripts.acquire_data
```

The checkpoint is `HuggingFaceTB/SmolVLM2-256M-Video-Instruct`, pinned revision
`067788b187b95ebe7b2e040b3e4299e342e5b8fd`. It is acquired without an account,
token or remote model code. `scripts/provenance.py` verifies the downloaded
safetensors SHA256 against the public LFS metadata. The public MVBench archive
and annotations are pinned to dataset revision
`230a2d4fac8900333c61754641c7a13e069ac9c6`. Only eight requested videos are extracted.

Expected local footprint is about **3 GB total**, including software, a 1.03 GB
FP32 checkpoint, the 16.8 MB video archive and extracted clips. Downloads enforce
a 5,000,000,000-byte cap. Read actual final size in `results/provenance.json`.
RAM use is measured separately; model weights are loaded as BF16. This experiment
runs entirely on CPU. The RTX 5070 Laptop GPU is detected but its CUDA software
stack would exceed the storage budget during installation. GPU VRAM allocated
and reserved measurements are unavailable; no GPU performance gain is claimed.

## Reproduce the baseline and candidate

All commands below use a watchdog and write logs/status files. Each inference job
is finite; samples and repeats are capped. Weight loading is excluded from the
benchmark time boundary. Default settings are in `configs/default.json`.

```powershell
# Real baseline, persistent native Transformers cache, free-text sanity output
.\.runtime\python\python.exe -m scripts.run_bounded --seconds 600 --log results/raw/baseline_reproduced.log benchmarks.baseline

# One original or optimized multiple-choice QA request
.\.runtime\python\python.exe -m scripts.run_bounded --seconds 180 --log results/raw/inference_native.log src.inference --mode native --sample 0
.\.runtime\python\python.exe -m scripts.run_bounded --seconds 180 --log results/raw/inference_dynamic.log src.inference --mode dynamic --sample 0

# Unit tests, then real-checkpoint cache integration checks
.\.runtime\python\python.exe -m scripts.run_bounded --seconds 300 --log results/raw/unit_tests_reproduced.log pytest -q tests
.\.runtime\python\python.exe -m scripts.run_bounded --seconds 600 --log results/raw/integration_reproduced.log scripts.verify_integration

# Same 8 videos x 3 repeats x 4 configurations, plus excluded warmups
.\.runtime\python\python.exe -m scripts.run_bounded --seconds 1800 --log results/raw/benchmark_reproduced.log benchmarks.evaluate
.\.runtime\python\python.exe -m scripts.summarize
.\.runtime\python\python.exe -m scripts.plot_results
.\.runtime\python\python.exe -m scripts.write_paper
.\.runtime\python\python.exe -m scripts.render_paper
.\.runtime\python\python.exe -m scripts.check_pdf
.\.runtime\python\python.exe -m scripts.provenance
```

The complete benchmark takes several minutes on this CPU. Re-running it replaces
the default raw JSONL and derived reports; preserve a copy if comparing runs.
After regenerating the PDF, inspect every page image under `tmp/pdfs/` before
marking its visual review passed. `scripts.validate_results` is the final audit
of the delivered run and also requires the recorded final tests, CLI smoke check
and manual PDF review. Its saved result is `results/final_verification.json`.
Native, uniform 8-bit, uniform 4-bit and dynamic modes share the same model and
inputs. `benchmarks.memory_benchmark` and `benchmarks.throughput_benchmark` expose
memory and timing rows from that **same** measured run, avoiding inconsistent
denominators from separate experimental inputs.

## What is verified

- A pretrained video model encodes each frame only once. Its real historical
  visual K/V survive subsequent frame calls and are reused for QA.
- Native BF16 reference and custom 16-bit storage produce identical smoke-test
  answer logits. The model implementation and original weights are not patched.
- Packed 4/8-bit payloads, scales/offsets, binary shape headers and layer/frame
  descriptors are measured. Python bookkeeping and transient reconstructed
  attention buffers are separate from persistent tensor-storage accounting.
- Unit tests cover feature values, boundaries, nibble packing, error bounds,
  byte counts, mixed retrieval, positions and decoder attention compatibility.
  Real-model integration tests exercise the public checkpoint separately.
- The information-score trace comes from actual video observations, not invented
  signals. The heuristic can choose different precisions within one video.

The streaming harness is project code around SmolVLM2's supported causal cache;
it is not a port of ReKV. There is no window eviction, retrieval system, learned
calibration, rotation method or fused low-bit attention kernel. Reconstructed
history returns to native BF16 for attention. Compression can therefore reduce
persistent cache bytes while increasing end-to-end latency.

## Results and interpretation

Read `results/metrics.json` for units and uncertainty, `results/timeline.csv` for
frame decisions, and `results/figures/` for cache, throughput, accuracy, memory
tradeoff, overhead and timeline plots. GPU VRAM is an explicitly unavailable
panel, with CPU process RSS reported separately. Accuracy is based on eight
unique questions; repeats do not increase the quality sample size. This is a
diagnostic subset of MVBench action_antonym, **not official MVBench evaluation**.

`docs/model_selection.md`, `docs/architecture.md` and
`docs/implementation_notes.md` explain the source inspection, cache mapping,
quantization equations, feature scales, exact evaluation protocol and failures.
`docs/baseline_source/` preserves the tested baseline source and configuration;
`results/provenance.json` records source hashes and installed package versions.

## Sources and licenses

- [SmolVLM2 checkpoint](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct)
  and [Transformers](https://github.com/huggingface/transformers/tree/v4.51.3): Apache 2.0.
- [PyTorch](https://pytorch.org/): BSD-style; [OpenCV](https://opencv.org/license/): Apache 2.0.
- [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench): dataset card declares
  MIT; original video rights remain with their creators/platforms and the card
  limits video use to academic research. See `docs/mvbench_card.md`; no videos are
  redistributed as a published deliverable or uploaded to another service.
- [ReKV](https://github.com/Becomebright/ReKV) was inspected as an alternative,
  not executed or copied into this implementation.
- [KIVI](https://arxiv.org/abs/2402.02750) is related work, not an implemented method.

Public source snapshots and API metadata in `docs/` retain their upstream license
context. New project code is supplied for this prototype; no third-party model
weights or datasets are claimed as original work.
