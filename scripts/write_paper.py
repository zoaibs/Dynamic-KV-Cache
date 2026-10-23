"""Write the paper and final summary strictly from completed measurements."""
import json
from scripts.resources import ROOT

def main():
    data=json.loads((ROOT/'results/metrics.json').read_text())
    rows=data['metrics']
    by={r['mode']:r for r in rows}
    base,dynamic=by['native'],by['dynamic']
    env=json.loads((ROOT/'results/environment.json').read_text())
    counts=data['dynamic_precision_counts']
    table='| Configuration | Cache MiB | Accuracy % | Frames/s | Latency s |\n|---|---:|---:|---:|---:|\n'
    labels={'native':'Native BF16','8':'Uniform 8-bit','4':'Uniform 4-bit','dynamic':'Dynamic'}
    for r in rows:
        table+=(f'| {labels[r["mode"]]} | {r["cpu_cache_bytes_mean"]/2**20:.3f} | '
                f'{r["accuracy_percent"]:.1f} | {r["frames_per_second"]:.3f} | '
                f'{r["latency_seconds_mean"]:.3f} |\n')
    timing='| Configuration | FPS repeat SD | Latency repeat SD (s) | Pack ms | Unpack ms |\n|---|---:|---:|---:|---:|\n'
    for r in rows:
        timing+=(f'| {labels[r["mode"]]} | {r["frames_per_second_repeat_std"]:.4f} | '
                 f'{r["latency_repeat_std"]:.4f} | {r["pack_ms_mean"]:.1f} | {r["unpack_ms_mean"]:.1f} |\n')
    paper=f'''# Frame-Adaptive Packed KV Storage for Local Video Question Answering

An eight-video CPU feasibility study | 22 October 2026

## Abstract

We implement and evaluate a training-free storage scheduler for the actual temporal
key-value cache of a pretrained video-language model. OpenCV Canny edge density,
Farneback optical-flow magnitude and grayscale frame difference choose 4-, 8- or
16-bit storage independently for newly appended frames. Four-bit values are
physically packed; per-group scales, offsets, shape headers and frame mapping are
included in storage accounting. We use SmolVLM2-256M with an explicitly verified
causal frame-append harness and compare native BF16, uniform 8-bit, uniform 4-bit
and dynamic cache configurations on eight public MVBench action-antonym videos.
The dynamic cache reduces persistent CPU tensor storage by
{dynamic['cache_reduction_percent']:.1f}% relative to native, with an accuracy
difference of {dynamic['accuracy_delta_pp']:+.1f} percentage points on this subset.
Measured end-to-end CPU throughput changes by {dynamic['throughput_change_percent']:+.1f}%.
These small-sample measurements establish a working compression path, not a
general accuracy guarantee or GPU acceleration result. A 5 GB total workspace
budget prevents installing the much larger CUDA runtime. No model training or
synthetic inference results are used.

## 1. Introduction

A streaming video assistant should reuse its historical visual representations
when a new frame or question arrives. A persistent attention cache can provide
that reuse, but storage grows with temporal context. Lower-bit storage can reduce
the persistent footprint while adding packing, reconstruction and metadata
costs. Whether this improves end-to-end execution is therefore an empirical
question rather than a consequence of bit width alone.

This prototype asks a deliberately narrow question: can a simple, reproducible
frame score drive mixed-bit storage in a pretrained model's genuine video KV
path, and what happens on a small local workload? The contribution is a tested
engineering implementation and an auditable measurement record. We do not claim
a new pretrained streaming architecture, optimal thresholds or a competitive
MVBench score. All configurations share a checkpoint and causal input schedule.

## 2. Related Work

SmolVLM [1] studies compact multimodal models and supplies the public video-capable
checkpoint used here. ReKV [2] separates streaming video encoding from QA and
retrieves historical visual caches from memory tiers. Its CUDA-oriented code was
inspected but not run in this project. Our bounded append-only harness provides
no retrieval or sliding window. KIVI [3] analyzes asymmetric key/value
quantization and proposes a specialized low-bit design; our simpler contiguous
groupwise affine representation is not an implementation of KIVI. MVBench [4]
defines a broader temporal-understanding benchmark. We use just eight questions
from one of its tasks and explicitly deviate from its official evaluation.

<!-- PAGE BREAK -->

## 3. Methodology

### 3.1 Temporal reuse and baseline

We retain one Transformers Cache object across separate video-frame forward
calls. The unmodified pretrained vision encoder converts a new frame to visual
embeddings, which replace image-token positions before entering the causal text
decoder. Historical visual keys and values remain in the cache; the next frame
and the final question receive them through the actual attention Cache.update
interface. Absolute cache positions advance across calls, preserving RoPE
semantics. Earlier images are not re-encoded during QA. The native reference uses
unmodified Transformers DynamicCache. The streaming harness itself is project
code, not an upstream ReKV or SmolVLM streaming evaluation implementation.

![Schematic architecture, not an experimental result](../results/figures/architecture.png)

Figure 1. The native baseline substitutes DynamicCache for the packed-storage
and reconstruction path. The figure is a schematic of implemented components.

### 3.2 Information score

At 160x120 grayscale resolution, E_raw is the Canny edge-pixel fraction using
thresholds 100/200. M_raw is mean Farneback flow magnitude between successive
sampled observations, and D_raw is mean absolute grayscale difference / 255.
We normalize E=clip(E_raw/0.20,0,1), M=clip(M_raw/8,0,1) and
D=clip(D_raw/0.25,0,1). All three scale constants are explicit heuristics.

S = 0.3 E + 0.4 M + 0.3 D.

Assign 4 bits for S < 0.30, 8 bits for 0.30 <= S < 0.65, and 16 bits otherwise.
The first valid frame measures edges but sets motion/difference to zero. Missing
frames do not advance scoring state; benchmark decoding rejects missing samples.
State resets between videos. Weights, thresholds and scales are configurable;
there is no learned heuristic or feedback calibration.

### 3.3 Packed storage and retrieval

For each layer/frame K and V tensor, contiguous groups of 64 values use FP32
offset a=min(x) and scale s=(max(x)-a)/(2^b-1). Constant groups use s=1. We store
q=round((x-a)/s), clipped to the unsigned b-bit range, and reconstruct q*s+a.
Two 4-bit values share one uint8 byte. Each tensor includes a real 64-byte shape/
dtype header; each layer/frame pair adds a 32-byte position/precision descriptor.
Sixteen-bit segments preserve native BF16 bytes exactly. Both K and V use the
same simple grouping rule, with no outlier rotation or specialized attention kernel.

New K/V remain native for the current frame's attention and are stored at the
assigned precision for later calls. Each subsequent layer reconstructs historical
segments to native BF16, concatenates the current K/V and performs ordinary SDPA
attention. Text-only prefix and question segments remain BF16 in every compressed
mode. The frame bit assignment also covers its associated timestamp/marker tokens.

<!-- PAGE BREAK -->

## 4. Experimental Setup

Hardware is a Windows machine with AMD Ryzen AI 9 365 CPU, 20 logical processors,
{env['ram_total_bytes']/2**30:.1f} GiB RAM, and an NVIDIA RTX 5070 Laptop GPU with
8151 MiB VRAM (driver 592.82). All measured inference runs use four CPU threads,
PyTorch 2.7.1+cpu, Transformers 4.51.3, BF16 and seed 1729. The CUDA 12.8 Windows
wheel alone is 3.27 GB compressed, so it was not installed under the user's 5 GB
software-plus-data cap. This is whole-model CPU inference, not GPU execution
with an offloaded cache. GPU allocated/reserved peaks are unavailable.

The checkpoint is HuggingFaceTB/SmolVLM2-256M-Video-Instruct at revision
067788b187b95ebe7b2e040b3e4299e342e5b8fd. It has 30 decoder layers, three KV heads
and head dimension 64. We disable image splitting and process each frame at the
512-pixel video size, yielding 64 visual tokens. Pretrained weights are identical
across configurations; no dataset training or weight quantization is performed.

Inputs are the first eight published MVBench action_antonym records at dataset
revision 230a2d4fac8900333c61754641c7a13e069ac9c6. Four uniformly spaced frames,
including the first and last, are decoded from each public clip. A constrained
decoder chooses the highest-logit unspaced uppercase option label, then exact
candidate matching determines correctness. This differs from official MVBench
in subset, frame count, prompt and decoding. Full-data scores are not inferred.

We exclude one complete warmup request per configuration, then run three repeats
with cyclically rotated configuration order: 96 measured requests and eight
unique quality observations per mode. Latency covers decode through answer,
including preprocessing, scoring, packing and reconstruction, but excludes model
loading, garbage collection and logging. Throughput is total sampled frames /
total request time within each repeat. Standard deviations describe the three
repeat-level means. Accuracy intervals use eight unique questions, not repeats.

## 5. Results

{table}

Table 1. Mean persistent CPU cache MiB after the last frame and before QA,
including prefix, frame markers, packed payload and binary metadata. Accuracy
uses eight unique videos. Python bookkeeping, allocator padding and transient
reconstructed attention tensors are outside this persistent-storage metric.

![Observed cache storage and CPU throughput](../results/figures/paper_overview.png)

Figure 2. Storage includes conversion metadata. Throughput error bars show
sample standard deviation of repeat-level means; neither panel measures GPU speed.

<!-- PAGE BREAK -->

## 5. Results (continued)

The dynamic cache reduces persistent storage by {dynamic['cache_reduction_percent']:.1f}%
relative to native BF16 and changes throughput by {dynamic['throughput_change_percent']:+.1f}%.
Its accuracy is {dynamic['correct']}/8 ({dynamic['accuracy_percent']:.1f}%), compared
with {base['correct']}/8 ({base['accuracy_percent']:.1f}%) for native. The dynamic
95% Wilson interval is [{dynamic['accuracy_wilson95_low']:.1f},
{dynamic['accuracy_wilson95_high']:.1f}]%. Identical correctness counts or zero
observed percentage-point loss in such a small subset cannot establish equivalence.
One changed answer would move accuracy by 12.5 percentage points, so this study
cannot resolve a 1.2-point target. Predictions are deterministic across repeats.

{timing}

Table 2. Repeat-level timing variability and mean per-request conversion cost.
Dynamic OpenCV scoring plus scheduling takes {dynamic['score_ms_mean']:.2f} ms per
complete request. Conversion timers exclude the final tensor concatenation and
ordinary attention arithmetic, which remain included in end-to-end time.

![Information-score and precision trace from real videos](../results/figures/precision_timeline.png)

Figure 3. The 32 real frame observations select {counts['4']} four-bit,
{counts['8']} eight-bit and {counts['16']} sixteen-bit segments. Video boundaries
reset scoring state. Widths absent from the real trace are not fabricated; unit
tests and a separate real-model forced schedule exercise simultaneous 4/8/16
retrieval without treating those forced choices as benchmark observations.

GPU VRAM is explicitly unavailable. CPU process peak RSS is sampled every 10 ms
and includes weights, interpreter and transient buffers, so it is not comparable
to persistent KV bytes. Complete RSS, quality, memory/accuracy and overhead plots,
per-request logs and the scheduling CSV are provided with the project.

<!-- PAGE BREAK -->

## 6. Discussion and Limitations

The implementation demonstrates physical low-bit storage and reuse of actual
historical video K/V. It does not establish faster floating-point attention.
Unpacking restores native BF16 for the same attention kernel, and repeated
conversion and concatenation can erase any benefit from smaller persistent
storage. The measured throughput change must be interpreted in this CPU workload
and does not identify reduced memory traffic as a causal speedup mechanism.
No GPU allocator measurement or GPU throughput conclusion is available.

Dynamic choices are initial heuristics rather than optimized operating points.
The first frame's zero temporal features strongly favor low precision. Feature
normalization and sampling intervals affect choices, and the observed bit mix
does not show that all three thresholds are useful. The fixed-bit baselines are
essential comparators. Uniform 4-bit storage achieved a larger cache reduction
and slightly higher throughput than dynamic at the same observed correctness
count. This subset therefore provides no demonstrated benefit of the scheduler
over uniform 4-bit storage.

The eight-video subset is deterministic, tiny and restricted to one task. It is
not representative of long streaming videos, and repeats only characterize
timing variation. Four sampled frames may miss relevant motion. The pretrained
model and custom causal prompt can make incorrect or hallucinated predictions;
the native smoke-test multiple-choice answer is wrong. No accuracy is inferred
from a plausible-sounding description. The constrained letter protocol may
introduce additional answer bias. No streaming fine-tuning was performed.

Cache tensors reside exclusively in CPU RAM; no disk tier, retrieval index or
eviction policy is implemented. Streams remain bounded by the 8192-token context.
Persistent bytes exclude Python bookkeeping and transient dequantization buffers;
RSS includes these but its sampled peaks and allocator history limit precision.
The small cache and dominant model activations constrain conclusions about total
resident-memory savings. This prototype implements no RotateKV, learned scheduler,
attention-sink algorithm or fused quantized attention kernel.

Failures are preserved rather than hidden. The initial baseline preprocessing
call failed because the processor received duplicate size arguments; using its
processor properties fixed it. ReKV was inspected and rejected before execution
because its CUDA/FlashAttention-oriented stack did not fit this setup. CUDA
PyTorch was not downloaded under the storage cap. Initial installation and access
issues are documented separately and are not represented as model benchmark runs.

## 7. Conclusion

A compact pretrained video model can reuse real temporal K/V across frame calls
while storing different frames in packed low-bit formats. This study provides
working code, storage accounting, tests and honest local CPU measurements.
The evidence supports a bounded prototype; broader quality evaluation, longer
streams and GPU experiments require separate validation. The original resume
targets of 32% cache reduction, a 1.2 pp accuracy gap, 14% throughput improvement
and 18% lower VRAM are not promoted to verified general results by this study.

## References

[1] A. Marafioti et al. SmolVLM: Redefining small and efficient multimodal models.
2025. [arXiv:2504.05299](https://arxiv.org/abs/2504.05299).

[2] S. Di et al. Streaming Video Question-Answering with In-context Video KV-Cache
Retrieval. ICLR 2025. [arXiv:2503.00540](https://arxiv.org/abs/2503.00540).

[3] Z. Liu et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
ICML 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750).

[4] K. Li et al. MVBench: A Comprehensive Multi-modal Video Understanding Benchmark.
CVPR 2024. [arXiv:2311.17005](https://arxiv.org/abs/2311.17005).
'''
    (ROOT/'paper/paper.md').write_text(paper,encoding='utf-8')
    summary=f'''# Final summary

## Completed

Working local prototype: pretrained video inference, persistent visual cache reuse,
exactly the three requested OpenCV features, configurable heuristic scheduler,
physically packed 4-bit and 8-bit storage, lossless BF16 storage, per-layer/frame
mapping, real attention integration, unit/integration tests, controlled experiments,
raw logs, figures, documentation and a research-style paper. No model training,
paid service, personal input or system-wide configuration change was used.

Model: **HuggingFaceTB/SmolVLM2-256M-Video-Instruct**, revision
`067788b187b95ebe7b2e040b3e4299e342e5b8fd` (Apache 2.0). The streaming harness is
project code using the unmodified model and supported Transformers cache API.
This is not a port or benchmark of ReKV. Native BF16 is the baseline throughout.

All inference ran on the **AMD Ryzen AI 9 365 CPU**, using four threads and BF16.
RTX 5070 Laptop hardware was detected; GPU software was not installed because the
CUDA wheel alone was 3.27 GB compressed under the 5 GB total workspace cap. This
is CPU inference, not GPU inference with CPU offload. GPU peak allocated/reserved
memory and any GPU VRAM reduction remain **unmeasured**.

## Actual comparative measurements

{table}

Eight unique MVBench action_antonym videos, four sampled frames each, three
repeats per configuration: **96 measured requests**, plus four excluded warmups.
Same checkpoint, causal frame order, input set, prompt, seed and attention kernel.
Accuracy uses eight unique questions; repeats do not enlarge the quality sample.
This diagnostic protocol is **not official MVBench evaluation**.

Dynamic persistent CPU cache reduction versus native: **{dynamic['cache_reduction_percent']:.1f}%**.
Dynamic accuracy delta: **{dynamic['accuracy_delta_pp']:+.1f} percentage points** on
these eight questions. Dynamic CPU throughput change: **{dynamic['throughput_change_percent']:+.1f}%**.
Latency includes decode, preprocessing, vision/model inference, scheduling and
conversion through answer; it excludes model loading and log writing.

Uniform 4-bit storage saved more cache space and ran slightly faster than dynamic
with the same observed accuracy. This subset provides no evidence that the
dynamic policy is superior to uniform 4-bit storage.

{timing}

Timing SD is across three complete repetition-level means. Dynamic quality's
95% Wilson interval is [{dynamic['accuracy_wilson95_low']:.1f},
{dynamic['accuracy_wilson95_high']:.1f}]%. No statistical equivalence or general
performance gain is claimed. Original resume targets remain unsuitable as
unqualified verified results, especially GPU and full-benchmark claims.

## Verified behavior and accounting

The native cache retains historical VIDEO-frame keys across separate forward
calls. QA makes no additional vision encoder calls. Custom 16-bit storage gives
exactly the same smoke-test answer logits as native BF16. Packed caches directly
supply reconstructed K/V inside the real attention update path. A separate
real-model forced schedule tests simultaneous 4/8/16 retrieval. Dynamic benchmark
decisions across 32 observed frames are: **{counts['4']} at 4 bits, {counts['8']} at
8 bits, {counts['16']} at 16 bits**. Forced tests are not benchmark observations.

Primary bytes count persistent CPU tensor payload, scales, offsets, 64-byte
headers and 32-byte layer/frame descriptors, before adding the question. Prefix
and frame-associated markers are included. Python object bookkeeping, allocator
rounding and transient reconstructed buffers are excluded; sampled process RSS
includes them and the loaded model. GPU cache placement and disk KV storage are
zero by design; GPU allocator measurement fields are null/unavailable.

## Limitations and failures

- Small CPU-only diagnostic: eight questions, one task, four frames/video; 12.5 pp
  accuracy resolution cannot establish a 1.2 pp target. Native predictions can be wrong.
- Dequantization and concatenation return history to BF16 attention; persistent
  storage savings do not imply lower peak process memory or faster attention.
- Initial threshold heuristic; no calibration, rotation, specialized low-bit
  kernel, retrieval index, window eviction or foundation-model training.
- Append-only context limited to 8192 tokens; no long-video scalability claim.
- ReKV was inspected but not run. First preprocessing attempt failed; the log
  and successful correction remain in `results/raw/`. CUDA was not attempted
  after the package-size preflight ruled it out within the budget.

## Reproduction and artifacts

Use the project-local `.runtime/python/python.exe`; no system Python is required.
Exact commands and installation procedure: **README.md**. Main commands:

```powershell
.\\.runtime\\python\\python.exe -m scripts.run_bounded --seconds 300 --log results/raw/tests_reproduced.log pytest -q tests
.\\.runtime\\python\\python.exe -m scripts.run_bounded --seconds 1800 --log results/raw/benchmark_reproduced.log benchmarks.evaluate
.\\.runtime\\python\\python.exe -m scripts.summarize
.\\.runtime\\python\\python.exe -m scripts.plot_results
.\\.runtime\\python\\python.exe -m scripts.write_paper
```

- `src/`, `configs/`, `tests/`: implementation and reproducible settings.
- `results/environment.json`, `results/provenance.json`: hardware and hashes.
- `results/metrics.csv`, `results/metrics.json`: comparative measurements.
- `results/raw/benchmark.jsonl`: every measured request and frame decision.
- `results/raw/baseline_smoke.json`, `results/raw/integration.json`: real-model evidence.
- `results/raw/unit_tests.log`: unit test outcome; final verification logs are nearby.
- `results/timeline.csv`, `results/figures/`: actual-data traces and plots.
- `docs/`: model rationale, architecture, protocol, failures, source/license context.
- `docs/baseline_source/`: original measured baseline and configuration.
- `data/manifest.json`: exact public questions, video identities and hashes.
- `paper/paper.md`, `paper/paper.pdf`: research-style report; PDF rendering checks under `tmp/pdfs/`.

Project storage remains below the 5,000,000,000-byte cap; the exact final byte
count is recorded in `results/provenance.json` and final verification output.
'''
    (ROOT/'FINAL_SUMMARY.md').write_text(summary,encoding='utf-8')
    print('Paper Markdown and final summary generated from actual measured metrics')

if __name__=='__main__':
    main()
