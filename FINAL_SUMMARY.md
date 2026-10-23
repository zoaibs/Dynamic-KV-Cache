# GPU upgrade: measured results

The project now runs the pretrained video model and physically packed KV cache on the RTX 5070 Laptop GPU. The main attention kernel unpacks only the tile it consumes; it never reconstructs the entire historical cache. The earlier CPU experiment is preserved in `archives/cpu_v1.zip` and the original result files.

## Held-out quality

The frozen policy was selected on 36 validation videos, then evaluated on **72 different videos from two MVBench tasks**, requesting up to 16 distinct real frames. Seventy-one held-out clips yield 16; one yields 5 (mean 15.85). One validation clip yields 15. Frames are never duplicated to reach a count. Each video's three or four cyclic answer rotations form one statistical unit. Accuracy means choosing the ground-truth action by average conditional log-likelihood of its complete candidate string.

| Cache mode | Accuracy % | Delta pp | Used MiB | Allocated MiB |
|---|---|---|---|---|
| Native BF16 | 62.50 | +0.00 | 48.9 | 48.9 |
| Engine BF16 | 62.50 | +0.00 | 48.9 | 69.1 |
| Uniform 4-bit | 62.50 | +0.00 | 15.7 | 24.8 |
| Uniform 8-bit | 62.50 | +0.00 | 27.8 | 40.9 |
| Tuned dynamic | 63.89 | +1.39 | 19.8 | 27.7 |
| Original dynamic | 62.50 | +0.00 | 20.7 | 28.8 |
| Random precision | 62.50 | +0.00 | 19.8 | 26.4 |

Dynamic versus native: **+1.39 percentage points**, paired 95% video-bootstrap interval **[+0.00, +4.17]**. Answer agreement is 98.61%. Candidate log-likelihood scores change (RMSE 0.2543, maximum absolute change 1.2778). The no-video control obtains 37.50%. Equal top-choice accuracy would not mean identical scores or lossless compression. This sample cannot establish general accuracy equivalence.

Used cache storage falls **59.5%**; actual allocated cache buffers fall **43.4%**. Both include quantization metadata. Allocated capacity also includes growth slack and reserved QA space. These are cache savings, not whole-model VRAM savings.

## End-to-end GPU throughput

| Frames | Native fps | Dynamic fps | Change | Native / dynamic capacity MiB |
|---|---|---|---|---|
| 4 | 4.06 | 4.47 | +9.9% | 12.8 / 9.6 |
| 16 | 6.47 | 6.92 | +7.0% | 49.3 / 28.1 |
| 32 | 7.13 | 7.97 | +11.8% | 98.1 / 41.2 |
| 64 | 8.16 | 8.81 | +8.0% | 195.6 / 79.4 |

Measured on three real validation clips, three repeats, five modes: **180 complete requests**. Timing includes decode, preprocessing, dynamic features, batched vision, sequential frame appends, packing, online growth and complete candidate-scored QA. CPU features overlap independent GPU vision work; residual waits are included. The earlier serial-feature timing run is preserved under `results/raw/gpu_timing_serial*`. Model loading and excluded JIT warmups are outside timing. This is offline video throughput, not live-camera arrival latency or autoregressive output-token/s.

At 64 frames dynamic changes throughput **+8.0%** versus native BF16 and **-2.8%** versus the new engine's BF16 control. Combined projections, normalization execution, attention and allocation changes contribute to the native comparison. Compression alone cannot receive credit for all of that gain. Absolute speed varied across separately scheduled runs; comparisons here use matched controls within the final run, not a serial-versus-overlap causal A/B.

## Policy experiments and implementation

Two checkpoints and five answer protocols were compared using validation data. The selected 500M checkpoint/protocol achieves 63.89% validation accuracy, versus 33.33% without video. All GPU cache comparisons use that same checkpoint; comparisons against the old 256M CPU accuracy would confound model, protocol, device and dataset changes.

The policy search tested **15 configurations** (12 dynamic plus three fixed controls). All tied on validation top-choice accuracy, so the search does **not** demonstrate an accuracy gain from different weights. Memory decided the dynamic winner: weights (0.2, 0.4, 0.4), validation-fitted scales, thresholds (0.55, 0.90), and an 8-bit first-frame floor. It matched native validation accuracy with 59.3% used-byte and 41.7% allocated-byte savings. It is the best eligible dynamic policy in this bounded search, not a globally optimal rule. Uniform 4-bit is a separate control and can save more memory. The original rule and deterministic random precision are reported as ablations, with no held-out retuning.

The GPU allocator starts with small pools and grows online, without future precision knowledge. Four-bit payloads pack two values per byte. K/V stay on the GPU. Full-history reconstruction is a diagnostic-only path. Independent vision embeddings are computed once, frame-only LM-head work is omitted in all modes, and QA branches reuse the video prefix. Security settings remain unchanged; a blocked experimental launcher was replaced with the earlier tested API and prebuilt CUDA operations.

On one 64-frame profile, matrix-multiply calls fall from 15235 to 8707 and concatenations from 8712 to 4424. GPU profiler timestamps are unavailable because CUPTI rejects this device; isolated packing/unpacking device time is not claimed. Synchronized complete-request timing and GPU memory counters remain available.

## Review and reproduce

- [README](README.md): runtime commands, file map and measurement boundaries.
- [GPU metrics](results/gpu_metrics.json), [quality table](results/gpu_quality.csv), [timing table](results/gpu_timing.csv).
- [Frozen configuration](configs/gpu_selected.json), [validation search](results/scheduler_validation.json), [accuracy protocol audit](results/accuracy_audit_500m_v3.json).
- [Raw quality](results/raw/gpu_test_quality.jsonl), [raw timing](results/raw/gpu_timing.jsonl), [figures](results/figures).
- [Manuscript](paper/paper.md), [six-page PDF](paper/paper.pdf), [final audit](results/gpu_verification.json), [provenance](results/gpu_provenance.json).

Limits: 72 held-out clips, two tasks, batch one, fixed architecture and 8192-token maximum; no eviction, concurrent serving or formal non-inferiority study. Timing variability reflects three repeats on the same clips. Whole-request visual activations can dominate peak VRAM, so cache reductions need not reduce total peak memory. The project and all public downloads stay local and below 15 GB.
