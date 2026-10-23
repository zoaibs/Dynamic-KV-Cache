# Model selection and verified cache behavior

Historical CPU-v1 notes follow. The current 500M CUDA model selection and verified cache path are described in [GPU design](gpu_design.md); exact GPU outcomes are in [the final summary](../FINAL_SUMMARY.md).

Selected checkpoint: **HuggingFaceTB/SmolVLM2-256M-Video-Instruct**, revision
`067788b187b95ebe7b2e040b3e4299e342e5b8fd`. Public weights are 1,025,998,224 bytes
in FP32 safetensors. All experiments load the same checkpoint as BF16; no weights
are trained or changed between configurations. Checkpoint license: Apache 2.0.

- Model card: https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct
- Research: https://arxiv.org/abs/2504.05299
- Inspected model implementation (Apache 2.0): https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/smolvlm/modeling_smolvlm.py
- Text attention/cache implementation: https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/llama/modeling_llama.py
- Dependency source tag: Transformers **v4.51.3**. Installed package hashes and
  upstream tag resolution are recorded in the provenance artifacts.

Architecture from the pinned configuration: SigLIP-style vision transformer,
768-dimensional vision states and 16-pixel patches; pixel-shuffle connector with
factor 4; Llama-family text decoder with 30 layers, hidden size 576, 9 query heads,
3 KV heads, head dimension 64 and maximum context 8192. A 512-pixel frame produces
64 visual tokens. The harness disables image splitting and uses the video-size
512-pixel preprocessing setting for each appended frame.

## Why this model

This is the smallest inspected publicly available video-capable candidate that
fits comfortably within the user-imposed 5 GB **total project** storage budget.
The public video checkpoint and Transformers implementation support reusable
`past_key_values`. Source inspection shows image embeddings replace image-token
positions before the causal text decoder; the decoder consumes and returns the
same cache. Consequently, those historical keys and values include VIDEO frame
representations, rather than only generated text.

The selected upstream model is a video model with a generic causal cache. **The
frame-append streaming harness is implemented in this project**; it is not an
upstream online-video service or ReKV implementation. The original unmodified
Transformers model and native `DynamicCache` are the reference baseline. All
configurations use identical causal frame append boundaries and question prompts.
The benchmark does not compare an upstream ReKV baseline against a different model.

## Rejected first candidate

[ReKV](https://github.com/Becomebright/ReKV), inspected commit
`1fd9a3dbf5dbff7f27069ae2f4463674c495e830`, was considered first. Its cache manager
explicitly creates CUDA streams/events and GPU-resident buffers, while its setup
depends on FlashAttention and describes a Linux/CUDA environment. Its temporal
cache is genuinely reusable and can reside in GPU, CPU RAM and disk. Under this
machine's Windows and 5 GB software-plus-weights budget, porting that stack was a
worse starting point than the smaller supported model. Its code was inspected,
but it was **not installed, run or benchmarked**. The repository's package metadata
declares Apache Software License; no standalone root LICENSE was present in the
inspected tree. No ReKV implementation code is incorporated in this prototype.

## Hardware feasibility

The machine has an RTX 5070 Laptop GPU with 8151 MiB VRAM and about 31 GiB RAM.
The official Windows PyTorch 2.7.1 CUDA 12.8 wheel is 3,273,066,072 bytes
compressed. Installing it entails extraction plus temporary wheel storage, in
addition to the 1.03 GB checkpoint and other dependencies. The project therefore
uses the 216 MB CPU wheel and CPU inference. This is a whole-model CPU run, **not
GPU inference with CPU cache offload**. No GPU throughput or VRAM reduction is
established. The selected checkpoint itself would fit GPU VRAM in a different
software environment; the storage budget constrains this run.

## Runtime evidence

`results/raw/baseline_smoke.json` verifies separate frame calls, persistent cache
growth, unchanged historical native keys and a question answered without further
vision encoder calls. `results/raw/integration.json` verifies custom 16-bit,
uniform 8-bit, uniform 4-bit and dynamic storage on the real checkpoint. Custom
16-bit and native BF16 answer logits agree exactly on the smoke input. Packed
low-bit history is reconstructed inside `Cache.update`, so it directly affects
the keys and values used by subsequent attention calls.

Every persistent tier in this run is **CPU RAM**. No video KV cache is written to
disk. Model weights, data and logs are stored on disk separately. No CUDA cache
allocation occurs; allocator peak GPU allocated/reserved fields are unavailable.

## Limitations

This harness has no sliding window, retrieval index or eviction policy. Its stream
is bounded by the decoder context, and the evaluation uses four frames per video.
The checkpoint was not trained or fine-tuned here for this frame-append harness.
Successful cache reuse does not establish high video-understanding quality.
Reported accuracy concerns only eight action-antonym questions and a constrained
letter decoder, not the model card's full MVBench score.
