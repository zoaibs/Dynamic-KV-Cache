# GPU implementation and experiment design

This is the current design. Older model-selection, architecture and implementation notes describe the CPU-v1 experiment, preserved in `archives/cpu_v1.zip`.

![GPU schematic](../results/figures/gpu_architecture.png)

The selected model is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, pinned revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Public FP32 checkpoint size is 2,029,990,624 bytes; inference weights are BF16. SHA256 is checked against the public model API's LFS record. The architecture has 32 decoder layers, 15 query heads, 5 KV heads and head dimension 64, with an 8192-token limit. It uses the same processor and frame-token formatting as the original reference. No remote model code or account is required. The checkpoint license is Apache 2.0.

The original 256M checkpoint remains installed. Validation compared both checkpoints and five answer protocols on 36 videos, selecting full-candidate conditional log-likelihood with the 500M model. Each final GPU comparison uses this checkpoint, BF16 weight values, identical video inputs and the same QA protocol. This change precludes treating the old CPU-v1 accuracy or latency as a matched baseline for the GPU study.

## Genuine cache reuse

The independent vision encoder computes each sampled image's features once, in batches up to eight for offline throughput. Those features replace the corresponding 64 image-token embeddings in one frame's decoder input. Decoder calls consume individual frames chronologically, with persistent cache state and advancing rotary positions. Video K/V produced in prior calls are consumed in later frame calls and in QA. Tests compare this direct text-model path with the original SmolVLM wrapper using exactly the same image embeddings, on both checkpoints, and check exact native K/V equality.

Native BF16 uses upstream DynamicCache and SDPA. Optimized attention dispatches only for the custom GPUCache. Combined QKV and gate/up projections preserve pretrained values exactly but change GEMM shapes and execution. CUDA-library normalization can differ in BF16 rounding. Engine BF16 is therefore an explicit control for execution differences. Fused BF16 is not claimed bit-identical to upstream model scores.

## Physical cache and attention

The allocator has CUDA pools for 4-bit uint8 nibbles, 8-bit uint8 values and BF16 values. Compressed 64-element head/token vectors have FP32 min/scale metadata, independently for K and V. One packing kernel per layer writes both. A GPU int32 row map and uint8 precision map identify each logical token's physical pool row. Prefix and QA tokens remain BF16; all tokens associated with a frame use that frame's selected storage bits.

Pools start at 64 rows and grow online, by at least 1.5x rounded to 64 rows. Growth copies stay on CUDA and are inside timing. No future precision sequence is provided to the production allocator. The map reserves 8192 tokens. Used-byte accounting and allocated-buffer accounting are separate; both count metadata, and allocated capacity includes map slack and QA reservation. Neither describes entire model peak VRAM.

Packed attention reads compressed historical tiles directly, dequantizes in registers and computes an online softmax with causal masking. The new append attends to its own uncompressed BF16 K/V; quantization affects later appends. It supports batch one, GQA and head dimension 64. Full-history reconstruction exists only for tests and a compatibility backend. There are no CPU KV payloads or disk KV files in the measured CUDA path.

QA branches execute sequentially on one CUDA stream. Packed branches share immutable video pools and write only reserved question rows. Native branches reuse the base cache and crop back to its original length after each candidate. This avoids an extra retained copy of the entire native video prefix while preserving repeated-question results. Concurrent branches are unsupported.

## Policy and accuracy

`src/policies.py` tests a bounded family of weights, thresholds and anchor variants. The winning dynamic rule has weights (0.2, 0.4, 0.4), validation-only 95th-percentile normalization, thresholds (0.55, 0.90) and an 8-bit first-frame floor. All 15 candidates tied on validation accuracy, so memory decided the winning dynamic policy. Uniform 4-bit remains a separate, potentially smaller alternative. The test set is not used for replacement-policy selection.

Video preparation decodes to EOF (bounded to 1024 source frames) and samples distinct available frames. OpenCV's WebM metadata and random seeking failed on some public clips; the actual failure is preserved in exploratory logs. No frames are invented to reach a requested count. The feature scorer operates at 160x120 and can omit features whose frozen weight is zero. The selected policy uses all three features.

Primary held-out accuracy uses 72 videos from two MVBench tasks, up to 16 distinct actual frames and all cyclic option rotations (240 questions per mode). Seventy-one held-out clips yield 16 frames; `action_antonym_148` has only 5, giving a mean of 15.85. Validation clip `action_antonym_081` yields 15. No frame is duplicated and no clip is dropped after inspecting answers. The candidate score is mean conditional token log-likelihood, and the predicted candidate is compared with the dataset ground truth. All variants share this protocol. A no-video control removes visual frames. Statistical resampling clusters answer rotations within a video and stratifies by task; the test set is not 240 independent videos.

## Measurement and limits

Performance uses three validation videos with enough actual frames, three repeats and 4/16/32/64 frames. Complete request timing includes decode, preprocessing, CPU features when consumed, vision, causal frame decoder appends, online packing/growth and full candidate QA. CPU feature work runs on one background worker while CUDA vision executes on the main thread. It uses the same sequential scorer and exact frame inputs; parity is checked in tests and validation-policy traces. Residual feature waits count as preparation; total feature runtime overlaps vision and must not be added again. The earlier serial-feature timing run is preserved. Loading and excluded warmups are outside timing. Whole-request and decoder/QA peak allocated/reserved VRAM are recorded separately; reserved allocator memory can depend on previous requests. Separate profiling records CPU dispatch and device time when available, without contaminating throughput measurements.

This is a streaming decoder adaptation of a pretrained video model, not a newly trained streaming model or an official benchmark submission. It does not extend the checkpoint context limit, implement eviction, support arbitrary head dimensions or provide production serving. The timing workload is recorded video processing, not live frame-arrival latency. No free-text token-generation throughput is claimed.

An experimental launcher variant was blocked by Windows Application Control. It was abandoned without changing security settings; the delivered implementation uses the earlier working launcher interface and prebuilt PyTorch operations. GPU Torch and Triton are isolated inside this workspace; all compute is local. Details and actual limitations belong in the final measured summary and raw logs.

The delivered 64-frame profile has 15,235 native versus 8,707 optimized matrix-multiply calls, and 8,712 versus 4,424 concatenations. CUPTI returns `CUPTI_ERROR_INVALID_DEVICE`, so CPU counts are available but device timestamps are not. Packing/unpacking device time is left unavailable rather than estimated from these CPU records. The synchronized end-to-end benchmark and PyTorch GPU memory measurements do not require CUPTI.
