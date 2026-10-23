# Which results to use

The delivered GPU study uses `gpu_metrics.json`, `gpu_quality.csv` and `gpu_timing.csv`. Its raw inputs are `raw/gpu_test_quality.jsonl` and `raw/gpu_timing.jsonl`; each phase has a frozen configuration snapshot with its SHA256. `gpu_verification.json` checks the complete experiment and `gpu_provenance.json` records the final source, checkpoint and runtime identities.

The policy, normalization and answer protocol were selected using validation data only:

| File | Scope |
|---|---|
| `accuracy_audit_256m_v3.json`, `accuracy_audit_500m_v3.json` | 36 validation videos; two checkpoints, five protocols; cyclic answer rotations and no-video controls |
| `scheduler_validation.json` | 36 validation videos; 15 precision policies; no held-out policy selection |
| `raw/scheduler_validation.jsonl` | Per-video scores, predictions and actual memory for every policy |
| `raw/gpu_test_quality.jsonl` | 72 held-out videos, seven modes, 240 rotated questions per mode |
| `raw/gpu_timing.jsonl` | Three validation clips, three repeats, four context sizes, five modes |
| `profiles/` | Separate profiled run; not a source for end-to-end throughput numbers |

Earlier `accuracy_audit_*` files without the `v3` suffix are pilot runs. The exploratory `raw/gpu_performance_pilot.json` used precomputed visual embeddings and a prior allocator; its timings are **not** the final complete-request measurements. `raw/gpu_kernel_benchmark.json` is a synthetic diagnostic, including dispatch gaps, and must not be described as pure hardware kernel latency or real video throughput.

`raw/gpu_timing_serial*` preserves the complete-request run with CPU features finished before vision. The final `raw/gpu_timing*` repeats the same inputs, modes and frozen policy while overlapping independent CPU feature computation with CUDA vision. Only residual feature waiting is additive to preparation time. Feature formulas, decisions and QA semantics are unchanged; this scheduling change was motivated by validation performance components, not held-out answer selection.

`raw/gpu_timing_interrupted*` is an excluded 66-request partial repeat. A filesystem byte-count audit overlapped its start, so the entire partial run was stopped and preserved, with the reason in `gpu_timing_interrupted_note.json`. No row from it contributes to final metrics. The final benchmark is restarted in full with other project compute idle. Its summaries read the exact final filename, never a glob of exploratory runs.

The GPU kernel correctness log uses independently computed attention as its reference. Final pytest also checks independent quantization error bounds, strided values, online growth, immutable prefix behavior, selected-checkpoint wrapper parity and long mixed-precision causal attention. It does not assert that low-bit model probabilities equal native BF16.

Failed experiments remain in place. Some initial tiles exceeded the GPU's shared-memory limit; a new launcher variant was blocked by Windows Application Control. The delivered implementation uses the earlier tested launcher interface and prebuilt PyTorch CUDA operations. No security configuration was changed. Consult each `.status.json` and its log rather than assuming every exploratory file represents a successful final run.

`metrics.json`, `metrics.csv`, `raw/benchmark.jsonl` and `final_verification.json` belong to the original eight-video, four-frame, 256M **CPU-v1** experiment. Their accuracy and latency are not directly comparable to the 500M GPU study, which changes checkpoint, protocol, datasets and context length. The complete original source/report snapshot is `../archives/cpu_v1.zip`.

The policy search found a tie in validation top-choice accuracy across all 15 candidates. Memory and score fidelity distinguished them; it did not show an accuracy gain from new heuristic weights. The test set is not reused to select a replacement policy after observing its answers.
