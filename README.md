# Dynamic KV Cache Scheduler: GPU edition

A local pretrained video-QA research prototype with real persistent KV storage at 4, 8 and 16 bits. The CUDA path packs two 4-bit values per byte and unpacks historical tiles inside causal attention. No model training, cloud compute, paid API or external judge is used.

Read [FINAL_SUMMARY.md](FINAL_SUMMARY.md) for measured outcomes, uncertainty and limitations. The main GPU tables are [quality](results/gpu_quality.csv) and [timing](results/gpu_timing.csv); the manuscript is [Markdown](paper/paper.md) and [PDF](paper/paper.pdf). The original CPU prototype, paper and results are preserved in `archives/cpu_v1.zip`. Its runtime remains usable; see [CPU-v1 instructions](docs/cpu_v1_readme.md).

## Use this workspace

PowerShell, from this workspace:

```powershell
# Auto-selects CUDA in the included GPU runtime
.\.runtime\gpu\python.exe -m scripts.run_bounded --seconds 180 --log results/raw/my_request.log src.inference --device cuda --mode dynamic --sample 0 --frames 16

# Same checkpoint and input using upstream native BF16
.\.runtime\gpu\python.exe -m scripts.run_bounded --seconds 180 --log results/raw/my_native.log src.inference --device cuda --mode native --sample 0 --frames 16

# Correctness tests (CPU runtime skips GPU-specific tests)
.\.runtime\gpu\python.exe -m scripts.run_bounded --seconds 300 --log results/raw/my_tests.log pytest -q tests
```

Other GPU modes are `fused16`, `4`, `8`, `original` and `random`. `fused16` controls for the new execution engine without low-bit KV. Requests use the selected 500M checkpoint and full-candidate likelihood protocol from [configs/gpu_selected.json](configs/gpu_selected.json). `--sample` indexes the original eight demo clips; the held-out dataset is separate. `--frames` accepts 1 through 96 actual distinct sampled frames, subject to the model's 8192-token context limit. There is no context extension or eviction.

The included isolated Python 3.11.9 environments are `.runtime/python` (CPU) and `.runtime/gpu` (CUDA). VS Code's default project interpreter points to the latter. The tested GPU stack is Torch 2.7.1+cu128, Transformers 4.51.3 and Triton-Windows 3.3.1.post21 on the RTX 5070 Laptop GPU. Package pins are in [requirements_gpu.txt](requirements_gpu.txt); legacy CPU pins remain in `requirements.txt`.

For a fresh checkout, review `scripts/setup.ps1`, then `scripts/setup_gpu.ps1`. The CPU setup supplies the portable base, original checkpoint and data; the GPU setup creates a separate runtime, installs official CUDA Torch and Triton, extracts official Python headers, and acquires the pinned 500M checkpoint and added public dataset. It changes no system PATH, execution policy, security configuration or power setting. A blocked experimental launcher was abandoned; the delivered API and prebuilt CUDA operations work with the existing policy. Do not disable security to reproduce it.

The full workspace is roughly 11.5 GB, below the authorized **15,000,000,000-byte** cap. Most space is CUDA libraries, two isolated runtimes and two public FP32 checkpoints. Videos occupy under 0.1 GB of archives plus selected extracted clips. Models run BF16 on the GPU. Downloads and bounded jobs check workspace size; caches and temporary files are directed into `.runtime`. Available SSD space does not directly accelerate inference.

## Reproduce the frozen evaluation

The delivered config was frozen before held-out inference. These commands replace the named raw GPU experiment files; retain copies for comparisons. GPU jobs should run sequentially so measurements do not compete for the device.

```powershell
.\.runtime\gpu\python.exe -m scripts.run_bounded --seconds 1800 --log results/raw/gpu_test_quality.log benchmarks.gpu_evaluate --phase quality
.\.runtime\gpu\python.exe -m scripts.run_bounded --seconds 1800 --log results/raw/gpu_timing.log benchmarks.gpu_evaluate --phase timing
.\.runtime\gpu\python.exe -m scripts.summarize_gpu
.\.runtime\gpu\python.exe -m scripts.plot_gpu
.\.runtime\gpu\python.exe -m scripts.write_gpu_report
.\.runtime\gpu\python.exe -m scripts.render_paper
.\.runtime\gpu\python.exe -m scripts.check_pdf
.\.runtime\gpu\python.exe -m scripts.gpu_provenance
.\.runtime\gpu\python.exe -m scripts.gpu_validate
```

The final audit additionally requires the delivered final test and CLI-smoke logs and a recorded visual review of every regenerated PDF page in `tmp/pdfs`. Inspect the current pages before updating `results/pdf_checks.json`; do not reuse an earlier review. Source/checkpoint hashes and the selected-config SHA must match the measured run. Separate operator profiles can be collected with bounded `scripts.profile_gpu`; profiling is excluded from throughput measurements.

## Measurement boundaries

Quality: 72 held-out videos (48 action-antonym, 24 fine-grained-action) from a pinned MVBench revision, disjoint by video hash from 36 validation clips and the original eight demos. Held-out refers to our selection; overlap with model pretraining is not audited. Sampling requests up to 16 distinct actual frames: 71 held-out clips yield 16 and one yields 5 (mean 15.85); one validation clip yields 15. Every mode uses the same available frames. Each cache mode answers all cyclic rotations of the three or four candidates: 240 questions per mode, clustered into 72 video units. Accuracy means the ground-truth action has the highest mean conditional token log-likelihood. Rotations do not create extra independent videos. A no-video control measures question/candidate bias. Confidence intervals use a task-stratified, paired video bootstrap with 10,000 resamples.

Timing: three validation videos, three repeats, 4/16/32/64 frames and five modes = 180 complete requests. Includes video decode, CPU preprocessing, scheduler features, vision encoding, all causal frame appends, GPU packing/growth, and full candidate-scored QA. CPU features overlap independent GPU vision; residual feature waits are counted as preparation. Their runtime is not additive to total latency. CUDA is synchronized at component boundaries. Model loading and excluded JIT warmups are outside the timer. The independent vision encoder batches up to eight recorded frames; the decoder consumes them chronologically. This is **offline recorded-video throughput**, not live-camera latency or generated-token/s. Timing variation is repeat-level SD over the same clips. The earlier serial-feature experiment remains under `results/raw/gpu_timing_serial*`.

Memory: used cache bytes include real payload, FP32 offsets/scales and used mapping entries. Allocated cache bytes include unused pool capacity, the full 8192-token map and QA reservation. Online pools grow without future precision knowledge. Whole-request and decoder/QA PyTorch peak allocated/reserved VRAM are recorded separately. Cache reduction is not the same as whole-model VRAM reduction.

All GPU cache comparisons share the same checkpoint, images, causal sequence, scoring and vision batching. Native mode uses upstream DynamicCache/SDPA. Optimized modes use combined QKV and gate/up projections, CUDA-library normalization and tiled packed attention. Engine BF16 isolates these execution changes from compression; gains versus native cannot all be attributed to low-bit storage. Original pretrained parameter values are unchanged and checked exactly when consolidating storage.

## Policy experiments

Model/protocol validation compared two SmolVLM2 checkpoints and five answer-scoring conventions. The final policy search tested 12 dynamic candidates plus uniform 4-bit, uniform 8-bit and Engine BF16. Alternatives changed weights, validation-fitted scales, thresholds, first-frame anchors and periodic BF16 anchors; two omit features. The best eligible dynamic policy maximizes validation accuracy, then used-cache saving, then score fidelity, requiring at least 25% used and allocated savings. No policy was tuned on held-out answers. The original heuristic and a deterministic random policy are final ablations. See [scheduler validation](results/scheduler_validation.json) and [model/protocol audit](results/accuracy_audit_500m_v3.json).

## Source and evidence map

| Path | Purpose |
|---|---|
| `src/streaming.py` | Shared CUDA model, vision batching, causal ingestion and QA |
| `src/gpu_kernels.py` | Real nibble packing and online-softmax attention consuming compressed history |
| `src/gpu_cache.py` | GPU precision pools, online growth, logical maps, accounting and sequential QA forks |
| `src/gpu_ops.py` | Exact-value projection consolidation and CUDA-library pointwise execution |
| `src/policies.py`, `src/selective_features.py` | Causal policy family and optional feature computation |
| `benchmarks/gpu_evaluate.py` | Frozen held-out evaluation and complete-request timings |
| `results/raw/gpu_*config.json` | Exact config and SHA at evaluation time |
| `results/gpu_metrics.json` | Derived results with uncertainty and measurement scope |
| `results/gpu_provenance.json`, `results/gpu_verification.json` | Source/model identity and final audit |
| `results/profiles/` | Separate operator counts and device timing, when available |
| `results/metrics.json`, `archives/cpu_v1.zip` | Preserved CPU-v1 evidence |

Limitations: inference only, batch one, 64-dimensional heads, this Llama-based decoder, no eviction or concurrent serving, and two tasks from one dataset. The policy is best only within a bounded validation search. Equal accuracy is not numerical equivalence. Public MVBench annotations are pinned; original video research-use terms still apply. The 500M checkpoint revision is `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; dataset revision is `230a2d4fac8900333c61754641c7a13e069ac9c6`.
