"""Generate the GPU manuscript and user summary from completed measurements."""
import json
from scripts.resources import ROOT

def main():
    metrics=json.loads((ROOT/'results/gpu_metrics.json').read_text())
    selected=json.loads((ROOT/'configs/gpu_selected.json').read_text())
    search=json.loads((ROOT/'results/scheduler_validation.json').read_text())
    profile=json.loads((ROOT/'results/profiles/summary.json').read_text())
    q={r['mode']:r for r in metrics['quality']}
    t={(r['mode'],r['requested_frames']):r for r in metrics['timing']}
    labels={'native':'Native BF16','fused16':'Engine BF16','4':'Uniform 4-bit','8':'Uniform 8-bit',
            'dynamic':'Tuned dynamic','original':'Original dynamic','random':'Random precision'}
    d=q['dynamic'];n=q['native'];long=t['dynamic',64];base=t['native',64];control=t['fused16',64]
    mib=lambda x:x/2**20
    quality_table='\n'.join(f"| {labels[m]} | {r['accuracy_percent']:.2f} | {r['native_delta_pp']:+.2f} | {mib(r['gpu_cache_used_bytes_mean']):.1f} | {mib(r['gpu_cache_allocated_bytes_mean']):.1f} |"
                            for m,r in q.items())
    timing_table='\n'.join(f"| {f} | {t['native',f]['frames_per_second']:.2f} | {t['dynamic',f]['frames_per_second']:.2f} | {t['dynamic',f]['throughput_change_percent']:+.1f}% | {mib(t['native',f]['gpu_cache_allocated_bytes_mean']):.1f} / {mib(t['dynamic',f]['gpu_cache_allocated_bytes_mean']):.1f} |" for f in [4,16,32,64])
    long_table='\n'.join(f"| {labels[m]} | {t[m,64]['latency_seconds']:.3f} | {t[m,64]['decoder_seconds']:.3f} | {t[m,64]['qa_seconds']:.3f} | {t[m,64]['frames_per_second']:.2f} |" for m in ['native','fused16','4','8','dynamic'])
    validation_table='\n'.join(f"| {r['policy']['name']} | {100*r['accuracy']:.2f} | {100*r['cache_reduction']:.1f}% | {100*r['allocated_cache_reduction']:.1f}% |" for r in search['summary'])
    engine_gain=100*(long['frames_per_second']/control['frames_per_second']-1)
    score_dci=f"[{d['delta_ci_low']:+.2f}, {d['delta_ci_high']:+.2f}]"
    paper=f'''# GPU Frame-Adaptive Packed KV Storage for Local Video Question Answering

Local inference study | RTX 5070 Laptop GPU | SmolVLM2-500M | Reproducible prototype

## Abstract

We replace a CPU packing prototype with GPU-resident 4/8/16-bit KV pools and causal attention that reconstructs compressed history inside its attention tiles. The system retains actual pretrained visual keys and values across frame calls, without training model weights. A bounded validation search selects the checkpoint, answer-scoring protocol and frame policy before held-out evaluation. On 72 disjoint videos from two MVBench tasks, tuned dynamic storage obtains {d['accuracy_percent']:.2f}% accuracy versus {n['accuracy_percent']:.2f}% for native BF16. The paired difference is {d['native_delta_pp']:+.2f} percentage points (95% video-bootstrap interval {score_dci}). At up to 16 frames, used cache bytes fall {d['used_cache_reduction_percent']:.1f}% and allocated capacity falls {d['allocated_cache_reduction_percent']:.1f}%. At 64 frames, complete-request throughput changes {long['throughput_change_percent']:+.1f}% versus native. Results apply to this checkpoint, hardware and bounded workload; they do not establish general accuracy equivalence or a universal speedup.

## 1. Motivation and scope

KV compression saves persistent storage but can add packing, reconstruction and dispatch overhead. Hugging Face explicitly notes that quantized caches can hurt latency with short contexts [1]. The earlier local CPU prototype used only eight clips, four frames and a small 7.2 MiB native cache. Its equal top-choice accuracy was a coarse, biased result, not evidence that quantization preserved model probabilities. That CPU experiment and its source are preserved in archives/cpu_v1.zip and its original result files.

This study makes four concrete changes: move inference and packed payloads to CUDA; consolidate decoder projection calls and reuse visual embeddings; fuse historical dequantization into attention; and test 4, 16, 32 and 64 real frames. Validation also compares two pretrained checkpoints and five answer protocols. The new GPU study is not an apples-to-apples speed or accuracy comparison with the old CPU run.

| Component | Delivered configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU, 8 GiB, SM 120 |
| Runtime | Python 3.11.9; PyTorch 2.7.1+cu128; Transformers 4.51.3; Triton 3.3.1 |
| Checkpoint | SmolVLM2-500M-Video-Instruct; BF16 weights, unchanged values |
| Decoder | 32 layers; 15 query / 5 KV heads; head dimension 64 |
| Context and batch | Checkpoint limit 8192 tokens; batch one; no context extension |
| Execution | Entirely local; CPU video preparation; CUDA vision, decoder and packed KV |

The independent vision encoder processes up to eight recorded frames per batch. Their embeddings then enter the causal decoder one frame at a time with advancing rotary positions. This measures offline recorded-video processing. It does not measure latency while waiting for a live camera frame.

<!-- PAGE BREAK -->
## 2. GPU implementation and controls

![GPU architecture schematic](../results/figures/gpu_architecture.png)

Figure 1. Implementation schematic, not a performance measurement. Vision embeddings feed chronological decoder appends; the policy controls packed storage and attention reads it directly.

### Physical representation

Each decoder layer stores K and V in separate 4-, 8- and 16-bit GPU pools, shared across the sequence through a logical token map. Four-bit storage packs two integer values into each uint8 byte; eight-bit storage uses uint8; sixteen-bit storage uses BF16. Each compressed head/token vector has 64 elements and an FP32 offset and scale. For b in {{4,8}}, scale=(max(x)-min(x))/(2^b-1), and integer values use clamped round-to-nearest-even affine quantization. A constant vector uses scale one. Keys are quantized after rotary embedding; values use the same per-token grouping. This is not KIVI's asymmetric per-channel key design [4].

One Triton pack launch writes both K and V for a decoder layer. No historical KV payload is copied to the CPU. The logical row index is int32, the precision code uint8. The scheduler's scalar feature decisions remain on the CPU. Prefix and question tokens stay BF16. Precision applies to all tokens in a frame segment, including its image tokens, timestamp and delimiters; it is not restricted to image tokens alone.

### Causal consumption without full-history reconstruction

A tiled online-softmax attention kernel [2,3] gathers historical precision codes and rows, loads packed K/V, and reconstructs only the tile in GPU registers. It applies grouped-query attention and an explicit bottom-right causal mask. The current append's K/V remain native BF16 while attending within that append; compressed storage affects later appends. The main path allocates no full-history reconstructed K/V tensor. A separate reconstruction function exists for correctness tests and compatibility only.

Each precision pool starts at 64 rows and grows online by at least 1.5x, rounded to 64 rows. The allocator does not know future frame precisions. Growth copies live pool storage on the GPU; these copies and packing are included in timing. The token map reserves the checkpoint's 8192-token capacity. Exact question-token capacity is reserved after the video. Used bytes include payload, scales, offsets and the used map; allocated bytes include unused pool capacity and the entire map. Python object overhead is excluded from these cache-only counters. Whole-request PyTorch allocated/reserved VRAM is recorded separately.

### Reducing dispatch and temporary tensors

Q/K/V projections become one GEMM, as do gate/up projections. Exact checks preserve parameter values in consolidated storage. Normalization and SiLU/multiply use prebuilt PyTorch CUDA operations, not custom fused pointwise kernels. Every mode reuses image embeddings and omits frame-only LM-head work. CPU features overlap independent CUDA vision; tests verify identical synchronous/background values and timing includes residual waits. Sequential QA branches share the video prefix; the native cache is cropped back after each candidate.

Native BF16 uses upstream Transformers DynamicCache and SDPA with its original projection/normalization execution. Engine BF16 uses the optimized engine and the same packed-attention implementation with sixteen-bit pools. It controls for execution changes independently of low-bit quantization. Any speedup versus native includes these execution changes, online cache allocation differences and compression. The Engine BF16 comparison is necessary before attributing a gain solely to quantization.

A separate 64-frame operator profile records {profile['native']['aten_mm_count']} native versus {profile['dynamic']['aten_mm_count']} optimized matrix-multiply calls, and {profile['native']['aten_cat_count']} versus {profile['dynamic']['aten_cat_count']} concatenations. Device timestamps are unavailable (CUPTI_ERROR_INVALID_DEVICE), so isolated packing/unpacking device time is not reported. Windows blocked an experimental launcher; the delivered earlier API works without security changes. Failed logs are retained.

<!-- PAGE BREAK -->
## 3. Validation, policy search and evaluation design

Public MVBench [6] is pinned to revision 230a2d4fac8900333c61754641c7a13e069ac9c6. Seeded splits exclude the original eight clips: 36 validation videos (24 action-antonym, 12 fine-grained-action) and 72 held-out videos (48 and 24). Paths and video hashes are disjoint. Sequential decoding samples up to 16 distinct actual frames evenly, without repetition. One validation clip has 15 frames; one held-out clip has 5, giving a held-out mean of {n['actual_frames_mean']:.2f}.

Validation compares 256M and 500M SmolVLM2 [5] with five protocols: three letter-token conventions, candidate likelihood, and candidate likelihood adjusted by a no-video prior. The selected 500M candidate protocol scores 63.89% versus 33.33% without video; the 256M counterpart scores 58.33% versus 52.78%. Selection uses all cyclic option rotations. Every final GPU mode shares the selected checkpoint and protocol.

The answer is the candidate with maximum mean conditional token log-likelihood, avoiding letter-token conventions. Cyclic rotations yield 120 validation and 240 held-out questions, clustered into 36 and 72 video units, not extra independent clips. No external judge is used. The no-video control retains the question and candidate strings.

### Bounded policy search

The search tests 12 dynamic policies plus three fixed controls. Edge density, Farneback flow and grayscale difference feed varied weights, thresholds and first-frame/periodic anchors; two variants omit features. Scales are validation-only 95th percentiles. Per-frame decisions use current and previous frames, without future-clip normalization. Scheduler validation scores one candidate order; held-out evaluation checks every rotation.

| Validation policy | Accuracy % | Used saving | Allocated saving |
|---|---|---|---|
{validation_table}

Among dynamic policies saving at least 25% in both used and allocated bytes, selection maximizes validation accuracy, then used-byte reduction, then native-score fidelity. All 15 candidates tied on top-choice validation accuracy, so memory decided the winner; this is not evidence of an accuracy gain from tuned weights. The winner uses weights (0.2, 0.4, 0.4), scales ({selected['policy']['scales'][0]:.5f}, {selected['policy']['scales'][1]:.5f}, {selected['policy']['scales'][2]:.5f}), thresholds (0.55, 0.90) and an 8-bit first-frame floor. The original rule is retained as an ablation. This bounded search does not establish global optimality. The frozen configuration includes the test-manifest hash and predates held-out inference.

<!-- PAGE BREAK -->
## 4. Held-out accuracy and cache storage

| Cache mode | Accuracy % | Delta pp | Used MiB | Allocated MiB |
|---|---|---|---|---|
{quality_table}

Table 1. Means over 72 held-out videos, requesting 16 frames (71 clips yield 16, one yields 5). Cache snapshots precede QA. Prefix/question tokens remain BF16. Random precision draws from validation precision frequencies with a deterministic per-video seed, matching the distribution in expectation rather than each clip's exact byte budget.

![Held-out quality](../results/figures/gpu_quality.png)

Figure 2. Accuracy uses the video as the statistical unit. Bars show 95% task-stratified bootstrap intervals from 10,000 resamples. The dashed line is the same-protocol no-video control ({n['no_video_accuracy_percent']:.2f}%).

Tuned dynamic accuracy is {d['accuracy_percent']:.2f}% with interval [{d['accuracy_ci_low']:.2f}, {d['accuracy_ci_high']:.2f}]%. Its paired difference from native is {d['native_delta_pp']:+.2f} pp with interval {score_dci}. Top-choice answer agreement with native is {d['answer_agreement_percent']:.2f}%. Candidate log-likelihood scores have RMSE {d['candidate_score_rmse']:.4f} and maximum absolute change {d['candidate_score_max_abs_delta']:.4f} versus native. Matching accuracy does not mean matching keys, values or probabilities: different predictions can offset, and score changes often do not cross a decision boundary.

The dynamic policy saves {d['used_cache_reduction_percent']:.1f}% of used bytes and {d['allocated_cache_reduction_percent']:.1f}% of physically allocated cache capacity. The smaller capacity saving reflects online growth, initially reserved pools, mapping capacity and QA space. We report both rather than describe theoretical low-bit payload ratios as measured VRAM savings. Peak whole-model memory is not reduced by this same percentage because weights and visual activations remain present.

The two selected tasks and small local sample do not represent all 20 MVBench tasks. Bootstrap uncertainty is conditional on this fixed checkpoint, policy and sampled task mixture. There is no formal non-inferiority conclusion and no guarantee of accuracy preservation on other videos.

<!-- PAGE BREAK -->
## 5. GPU throughput and longer contexts

| Frames | Native fps | Dynamic fps | Change | Native / dynamic capacity MiB |
|---|---|---|---|---|
{timing_table}

Table 2. End-to-end sampled frames per second. Each context has three real validation clips, three repetitions and five cache configurations (180 measured requests). Clip choice used task membership and sufficient real frames, not answer correctness. Configuration order rotates across requests. Warmups and model loading are excluded.

![Throughput and physical capacity](../results/figures/gpu_summary.png)

Figure 3. Throughput includes decode, CPU preparation/features, vision, causal frame appends, packing/growth, and complete candidate QA. CPU features overlap vision; unhidden waits count as preparation, so feature runtime is not additive. Error bars show sample SDs of three repetition-level means on the same clips, not population uncertainty across arbitrary videos.

| 64-frame mode | Total s | Decoder s | QA s | Frames/s |
|---|---|---|---|---|
{long_table}

At 64 frames, dynamic throughput changes {long['throughput_change_percent']:+.1f}% relative to native and {engine_gain:+.1f}% relative to Engine BF16. The native comparison is not a compression-only gain. Longer input raises traffic and attention compute; low-bit decode arithmetic and mixed-pool loads remain. Absolute speeds varied across separately scheduled runs, so serial-versus-overlap timings are not an isolated causal A/B. Final claims use matched controls within the final run.

Timing synchronizes CUDA at component boundaries. Each repetition-level throughput is total actual sampled frames divided by total request seconds across its three clips. Full candidate scoring is teacher-forced likelihood evaluation, not free-text token generation; these frames/s and QA/s are not decode-token/s. Timing uses validation clips, whereas primary quality uses held-out clips. The 64-frame native cache averages {base['cached_tokens_mean']:.0f} tokens and {mib(base['gpu_cache_used_bytes_mean']):.1f} MiB, substantially larger than the original 7.2 MiB CPU experiment. All inputs stay within the pretrained context limit.

<!-- PAGE BREAK -->
## 6. Limits, reproducibility and next experiments

### Related work

FlashAttention [2] and Triton's attention tutorial [3] motivate tiled attention without materializing attention matrices. KIVI [4] studies asymmetric key/value quantization; our simpler per-token affine groups are a different design. SmolVLM [5] supplies the pretrained weights and MVBench [6] supplies public video questions. None of their published speed or accuracy figures is adopted as a result here.

This prototype supports inference only, batch one, 64-dimensional heads and this Llama-based decoder. It does not implement paged serving, concurrent QA, eviction, beam search, gradients or arbitrary architectures. Prefix and QA storage remain BF16; a frame's precision is fixed at ingestion. This streaming adaptation is not a streaming-trained model or an official benchmark run. Held-out means excluded from our selection; overlap with model pretraining is not audited.

Policy calibration is limited to 36 validation clips and a hand-designed candidate family. Better accuracy research would broaden task and clip coverage, compare per-channel keys with per-token values, and fit a scheduler to score distortion or uncertainty while retaining a separate validation split. Better speed research would specialize homogeneous precision tiles, tune GQA sharing and tile sizes, reduce online growth copies, and evaluate near the 8192-token checkpoint limit. Those are future experiments, not delivered claims. The present study prioritizes a measured GPU path, real physical packing and transparent controls.

The workspace retains separate CPU and CUDA environments, both checkpoints, pinned public data, failed exploratory logs and the CPU-v1 archive. It remains below the user-authorized 15 GB cap. Model weights are loaded as BF16 but downloaded in their public FP32 checkpoint form. No cloud credits, paid services, private accounts, system-wide installations, security changes or destructive cleanup are used.

The final audit checks hashes, split separation, policy freeze consistency, row counts, per-frame precision, physical bytes, GPU tests, CLI output and PDF visual inspection. Exact environment, source and checkpoint identities are in results/gpu_provenance.json; the audit is results/gpu_verification.json. Raw answers and timings are in results/raw/gpu_test_quality.jsonl and gpu_timing.jsonl, with CSV tables in results/. Original CPU evidence remains results/metrics.json and archives/cpu_v1.zip.

### Conclusion

GPU-resident packed storage and attention that consumes compressed tiles provide a working alternative to CPU packing and full-history reconstruction. Measured accuracy, allocated capacity and complete-request throughput must be assessed together; this bounded study does not establish a universally superior scheduler.

## References

[1] Hugging Face. [Transformers 4.51.3: KV cache strategies](https://huggingface.co/docs/transformers/v4.51.3/en/kv_cache). Quantization trades memory against overhead; short-context latency can worsen.

[2] Dao et al. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), 2022. Source of the tiled online-softmax formulation, not a benchmark result reproduced here.

[3] Triton. [Fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html). Implementation reference; the delivered mixed-pool kernel is local source.

[4] Liu et al. [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750), 2024. Related quantization research, not an implemented baseline.

[5] Marafioti et al. [SmolVLM: Redefining small and efficient multimodal models](https://arxiv.org/abs/2504.05299), 2025. Public 500M video checkpoint pinned to revision 7b375e1b73b11138ff12fe22c8f2822d8fe03467.

[6] Li et al. [MVBench: A Comprehensive Multi-modal Video Understanding Benchmark](https://arxiv.org/abs/2311.17005), 2024. Only two tasks are evaluated here; original source-video research-use terms apply.
'''
    (ROOT/'paper/paper.md').write_text(paper,encoding='utf-8')
    summary=f'''# GPU upgrade: measured results

The project now runs the pretrained video model and physically packed KV cache on the RTX 5070 Laptop GPU. The main attention kernel unpacks only the tile it consumes; it never reconstructs the entire historical cache. The earlier CPU experiment is preserved in `archives/cpu_v1.zip` and the original result files.

## Held-out quality

The frozen policy was selected on 36 validation videos, then evaluated on **72 different videos from two MVBench tasks**, requesting up to 16 distinct real frames. Seventy-one held-out clips yield 16; one yields 5 (mean {n['actual_frames_mean']:.2f}). One validation clip yields 15. Frames are never duplicated to reach a count. Each video's three or four cyclic answer rotations form one statistical unit. Accuracy means choosing the ground-truth action by average conditional log-likelihood of its complete candidate string.

| Cache mode | Accuracy % | Delta pp | Used MiB | Allocated MiB |
|---|---|---|---|---|
{quality_table}

Dynamic versus native: **{d['native_delta_pp']:+.2f} percentage points**, paired 95% video-bootstrap interval **{score_dci}**. Answer agreement is {d['answer_agreement_percent']:.2f}%. Candidate log-likelihood scores change (RMSE {d['candidate_score_rmse']:.4f}, maximum absolute change {d['candidate_score_max_abs_delta']:.4f}). The no-video control obtains {n['no_video_accuracy_percent']:.2f}%. Equal top-choice accuracy would not mean identical scores or lossless compression. This sample cannot establish general accuracy equivalence.

Used cache storage falls **{d['used_cache_reduction_percent']:.1f}%**; actual allocated cache buffers fall **{d['allocated_cache_reduction_percent']:.1f}%**. Both include quantization metadata. Allocated capacity also includes growth slack and reserved QA space. These are cache savings, not whole-model VRAM savings.

## End-to-end GPU throughput

| Frames | Native fps | Dynamic fps | Change | Native / dynamic capacity MiB |
|---|---|---|---|---|
{timing_table}

Measured on three real validation clips, three repeats, five modes: **180 complete requests**. Timing includes decode, preprocessing, dynamic features, batched vision, sequential frame appends, packing, online growth and complete candidate-scored QA. CPU features overlap independent GPU vision work; residual waits are included. The earlier serial-feature timing run is preserved under `results/raw/gpu_timing_serial*`. Model loading and excluded JIT warmups are outside timing. This is offline video throughput, not live-camera arrival latency or autoregressive output-token/s.

At 64 frames dynamic changes throughput **{long['throughput_change_percent']:+.1f}%** versus native BF16 and **{engine_gain:+.1f}%** versus the new engine's BF16 control. Combined projections, normalization execution, attention and allocation changes contribute to the native comparison. Compression alone cannot receive credit for all of that gain. Absolute speed varied across separately scheduled runs; comparisons here use matched controls within the final run, not a serial-versus-overlap causal A/B.

## Policy experiments and implementation

Two checkpoints and five answer protocols were compared using validation data. The selected 500M checkpoint/protocol achieves 63.89% validation accuracy, versus 33.33% without video. All GPU cache comparisons use that same checkpoint; comparisons against the old 256M CPU accuracy would confound model, protocol, device and dataset changes.

The policy search tested **15 configurations** (12 dynamic plus three fixed controls). All tied on validation top-choice accuracy, so the search does **not** demonstrate an accuracy gain from different weights. Memory decided the dynamic winner: weights (0.2, 0.4, 0.4), validation-fitted scales, thresholds (0.55, 0.90), and an 8-bit first-frame floor. It matched native validation accuracy with 59.3% used-byte and 41.7% allocated-byte savings. It is the best eligible dynamic policy in this bounded search, not a globally optimal rule. Uniform 4-bit is a separate control and can save more memory. The original rule and deterministic random precision are reported as ablations, with no held-out retuning.

The GPU allocator starts with small pools and grows online, without future precision knowledge. Four-bit payloads pack two values per byte. K/V stay on the GPU. Full-history reconstruction is a diagnostic-only path. Independent vision embeddings are computed once, frame-only LM-head work is omitted in all modes, and QA branches reuse the video prefix. Security settings remain unchanged; a blocked experimental launcher was replaced with the earlier tested API and prebuilt CUDA operations.

On one 64-frame profile, matrix-multiply calls fall from {profile['native']['aten_mm_count']} to {profile['dynamic']['aten_mm_count']} and concatenations from {profile['native']['aten_cat_count']} to {profile['dynamic']['aten_cat_count']}. GPU profiler timestamps are unavailable because CUPTI rejects this device; isolated packing/unpacking device time is not claimed. Synchronized complete-request timing and GPU memory counters remain available.

## Review and reproduce

- [README](README.md): runtime commands, file map and measurement boundaries.
- [GPU metrics](results/gpu_metrics.json), [quality table](results/gpu_quality.csv), [timing table](results/gpu_timing.csv).
- [Frozen configuration](configs/gpu_selected.json), [validation search](results/scheduler_validation.json), [accuracy protocol audit](results/accuracy_audit_500m_v3.json).
- [Raw quality](results/raw/gpu_test_quality.jsonl), [raw timing](results/raw/gpu_timing.jsonl), [figures](results/figures).
- [Manuscript](paper/paper.md), [six-page PDF](paper/paper.pdf), [final audit](results/gpu_verification.json), [provenance](results/gpu_provenance.json).

Limits: 72 held-out clips, two tasks, batch one, fixed architecture and 8192-token maximum; no eviction, concurrent serving or formal non-inferiority study. Timing variability reflects three repeats on the same clips. Whole-request visual activations can dominate peak VRAM, so cache reductions need not reduce total peak memory. The project and all public downloads stay local and below 15 GB.
'''
    (ROOT/'FINAL_SUMMARY.md').write_text(summary,encoding='utf-8')
    print('Updated GPU manuscript and measured summary.')

if __name__=='__main__':main()
