# Gemma 4 workload reproducibility on kaggle-vllm 0.3.0

`kaggle-vllm` 0.3.0 is a general-purpose offline vLLM inference/runtime SDK for the **exact** validated Kaggle L4×4/SM89 profile. Gemma 4 is one high-value workload and a required release-candidate acceptance gate. Competition agents, submission formats, and tools belong in separate application projects; they are not dependencies or APIs of the core SDK.

The current four-L4 allocation is competition-controlled. Do not assume that every Kaggle Notebook offers this profile; the acceptance session must explicitly have four L4s and Internet disabled.

## Immutable inputs

| Input | Version or identity |
| --- | --- |
| Small SDK | `kaggle-vllm 0.3.0`; wheel SHA256 `08ff056319dcfbf1a020272e447fae39964d347065550c5d415ebd10cb89f5e3` |
| Runtime-v3 | `kaggle-vllm-v030-l4x4-runtime-v3.tar.gz`; SHA256 `0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73` |
| Native upstream vLLM | `vllm-0.30.0-cp312-cp312-linux_x86_64.whl`; SHA256 `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8` |
| Hardware | Kaggle Linux x86_64, exactly four NVIDIA L4, compute capability 8.9/SM89; TP=4 |
| Python | CPython 3.12/cp312 for native runtime; local SDK development uses custom Python 3.11 |
| Torch/CUDA | Torch 2.13.0+cu129; Python user-space CUDA 12.9; Kaggle system compiler/toolkit CUDA 12.8 |
| Related runtime | Triton 3.7.1; TorchCodec 0.16.0+cu129; FlashInfer 0.6.18.post1 |
| Gemma model | `/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2` |

The P31 source build used `CMAKE_CUDA_ARCHITECTURES=89`, `TORCH_CUDA_ARCH_LIST=8.9`, and concurrency derived from `nproc --all` (48 in that session). Normal 0.3.0 bootstrap uses the already-built native wheel and does not compile vLLM. Kaggle's CUDA 12.8 system toolkit is not replaced by the cu129 Python closure.

## Offline sourcing and acceptance

Attach the small SDK acceptance package and the four existing corrected split-v2 Datasets under `/kaggle/input/datasets/waqasm86/`, plus the saved P31 Notebook Output and the Kaggle-provided Gemma Input Model. The 6.6 GB runtime-v3 archive is not required. The L4 acceptance is run with Internet disabled and no Hugging Face or Kaggle API download. The [production acceptance notebook](../kaggle-notebooks/kaggle_vllm_0_3_0_l4x4_final_acceptance.ipynb) and [runner](../scripts/production_acceptance_v030.py) verify every selected wheel/native hash and run strict bootstrap, doctor, NCCL, and Gemma semantics in a **brand-new** four-L4 session.

The Gemma helper uses `KaggleLLM.chat()` with one `AutoProcessor` (or tokenizer fallback) template rendering and `KaggleSamplingParams(temperature=0.0, max_tokens=64)`. It uses TP=4, `gpu_memory_utilization=0.86`, and `max_model_len=2048`. It records load/generation time, generated token count, meaningful text, backend diagnostic, and worker cleanup. P31 Step 20 already passed semantic generation; the new isolated SDK wrapper path still needs fresh acceptance. The earlier raw Step 17 newline-like output was not a semantic success.

P31 proved that the vLLM native wheel contains and imports built-in FlashAttention-2 for SM89. **Gemma selected `TRITON_ATTN`** in P31. Backend selection is model-specific, and the SDK does not force FA2. The competition acceptance runs no unrelated model; any real-model FA2 selection test is deferred to a later permitted L4 environment.

## Attribution and data boundary

This document contains no competition dataset rows, prompts from restricted evaluation material, private agent outputs, credentials, or model weights. The SDK/runtime archive contains no Gemma model. Preserve upstream vLLM attribution and review the [third-party inventory](../release/v0.3.0-third-party-notices.md) before redistributing wheels. The Gemma model's own license, usage conditions, and attribution must be reviewed and carried separately by whoever mounts or publishes it. Do not place runtime CUDA distributions in the historical `waqasm86/kaggle-vllm-models` repository.

This preparation records no competition submission or result. The old v0.1.x/v0.2.0 T4 history, releases, assets, native binaries, Qwen TP=2 model artifacts, and licenses remain intact. Any later 0.3.0 publication must add new files and releases without replacing those historical materials.
