# Implementation and measurement notes

Historical CPU-v1 notes follow. The current execution, online GPU allocation and measurement boundaries are in [GPU design](gpu_design.md).

## Information score

Exactly three features are evaluated on BGR-to-grayscale frames resized to
160x120 using area interpolation:

- `E_raw`: fraction of pixels detected by OpenCV Canny, thresholds 100 and 200.
- `M_raw`: mean Euclidean magnitude from Farneback flow between successive
  sampled grayscale frames, in pixels per sampled-frame interval. Parameters:
  pyramid scale .5, levels 3, window 15, iterations 3, poly_n 5, poly_sigma 1.2.
- `D_raw`: mean absolute grayscale difference divided by 255.

Normalized features are `E=clip(E_raw/.20,0,1)`,
`M=clip(M_raw/8,0,1)`, `D=clip(D_raw/.25,0,1)`.
The scale constants are documented engineering heuristics, not fitted values.
`S=.3E+.4M+.3D`. Choose 4 bits below .30, 8 bits from .30 inclusive to .65
exclusive, and 16 bits at or above .65. Everything is configurable in
`configs/default.json`; the core policy contains no learning or calibration.

For a first valid frame, edge density is measured and motion/difference are zero.
Its score therefore cannot exceed .30 under the default weights. Missing frames
return an explicit missing flag and zero score without advancing the previous
frame; the standalone scheduler would select 4 bits. The real benchmark rejects
missing decoded frames rather than inserting fabricated observations. Scorer
state resets between videos. Motion is not normalized by elapsed seconds because
the requested feature is magnitude between successive observations, not velocity.

## Experimental protocol

Dataset: OpenGVLab/MVBench, action_antonym annotations, first eight records in
their published order. Videos come from the public `ssv2_video.zip` (16.8 MB).
Only the eight requested files are extracted. Source revision, file SHA256s and
all questions/answers are in the dataset manifest. No personal videos are used.

Each video supplies four distinct uniformly spaced indices from first to last
available frame. This is a small deterministic diagnostic subset of one task,
not official MVBench evaluation. It uses fewer frames and a different streaming
prompt/decoding method than the official protocol. Candidate letters A/B/C are
each one tokenizer token; the answer is the maximum next-token logit among those
unspaced uppercase labels after `Assistant:`. Exact correctness compares the
selected candidate with the original answer string. No external judge or API
scores answers. This constrained choice differs from free generation and may
bias results; the natural-language smoke description is not counted as accuracy.

Same model revision, BF16 weights, SDPA causal attention, seed 1729, four CPU
threads, inputs and sample order for all configurations. One complete request per
configuration is warmed up and excluded. Three measured repetitions follow;
configuration order is cyclically rotated by sample and repeat. The dataset
contains eight unique questions, not 24 independent quality observations.
Repeat-level time/throughput means and their sample standard deviations are
reported. Accuracy uses eight unique questions with 95% Wilson intervals; those
intervals should not be read as generalization guarantees for a selected subset.

End-to-end latency starts immediately before video decode and ends after the
answer is scored. It includes sampling, preprocessing, vision encoding, text
prefix, frame appends, dynamic scoring, packing and reconstruction. It excludes
weight loading, Python startup, garbage collection, warmup and log writing.
Frames/s = number of sampled frames divided by complete request latency; QA/s =
one divided by latency. Repetition throughput uses total frames/total time rather
than averaging per-request reciprocals. GPU synchronization is implemented for
a future compatible CUDA runtime; no GPU inference is claimed in this run.

Persistent cache is measured after the final frame and before adding QA text.
The primary storage denominator includes prefix and frame-associated markers,
plus packed headers, scales, offsets and layer/frame descriptors. A separate
frame-only field excludes the prefix. The original baseline has native BF16
tensor payload and no added quantization headers. Python object/container
bookkeeping and allocator rounding are excluded from these tensor-storage byte
counts. The process RSS measurement includes their impact, model weights and
temporary dequantized attention buffers; it samples every 10 ms and can miss very
short peaks. This is not a measurement of peak total cache-related allocation.

All persistent KV is in CPU RAM, with zero KV disk storage. The explicit zero
for GPU **cache placement** describes the CPU design; GPU allocator allocated and
reserved **measurements** are null/unavailable. The GPU figure labels this absence
instead of displaying invented zero measurements. CPU RSS and cache bytes have
different scopes and must not be interchanged.

Packing/reconstruction timers run inside real per-layer cache updates. They
include payload conversion, scale calculation and headers (packing), and history
reconstruction (unpacking). They exclude the final concatenation of tensors and
the model's ordinary attention arithmetic. OpenCV time covers all three features
and the threshold decision, excluding video decode. These timers are diagnostic
subintervals within end-to-end latency, not independent speed measurements.

## Failures and changes

- No Python or Git executable was available on the initial PATH. Installed an
  official portable CPython runtime entirely in `.runtime`; no global settings
  or system packages were changed. Git history is optional; hashes preserve the
  exact baseline, checkpoint and final sources.
- The sandbox initially denied system CIM queries and network requests. Hardware
  facts were obtained through NVIDIA's CLI and psutil; public downloads proceeded
  after the user enabled access. No private account or credential was used.
- A wheel-only dependency attempt could not resolve legacy `docopt`. Its source
  setup was inspected, then the tiny dependency was built in project-local temp
  storage. This was an installation failure, not a model experiment.
- The first baseline attempt failed because Transformers 4.51's processor passed
  `size` twice. Setting documented processor properties before calling it fixed
  preprocessing. Both the failed log and successful retry are preserved.
- Initial parallel dependency installation left duplicate metadata for the
  non-numerical `filelock` and `fsspec` packages. Their installations were
  reconciled after timing. The numerical stack and model/cache code were
  unchanged; final unit and real-model integration checks were rerun. The
  dependency lock records the final reproducible environment.
- ReKV was inspected but not executed. CUDA PyTorch was not downloaded because
  its compressed wheel plus extraction would breach the storage budget.

Failed actions and their actual status files remain under `results/raw/`; no
failure is replaced by a fabricated metric. Resume targets (32% cache reduction,
1.2 pp accuracy gap, 14% throughput gain, 18% VRAM reduction) remain unverified
until supported by an appropriately scoped measurement. A tiny CPU diagnostic
cannot establish any general GPU or full-benchmark resume claim.
