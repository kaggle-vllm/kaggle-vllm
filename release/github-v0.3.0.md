# kaggle-vllm 0.3.0 — draft release notes, local candidate

The exact P33 lean inference runtime is 168 SHA-verified dependency wheels plus the P31/P33/P35/P40-qualified vLLM 0.30.0 native wheel. Runtime-v3's 195-wheel closure is archival. The expanded runtime defaults to `/tmp/kaggle-vllm/v0.3.0/l4x4` because `/kaggle/working` had about 19.5 GiB writable space during qualification. Public artifacts remain unpublished; the immutable Hugging Face revision is intentionally unset. The exact built SDK wheel must pass the fresh final acceptance notebook before release claims are upgraded. Historical T4 profile data and the v0.2 command path remain available.

Python SDK for one validated Kaggle profile: Linux x86_64, CPython 3.12/cp312, exactly four NVIDIA L4/SM89 GPUs, TP=4. The SDK remains lightweight and imports without torch or vLLM. Its public wrapper exposes `KaggleLLM`, `KaggleSamplingParams`, `.generate()`, `.generate_text()`, `.chat()`, and `.upstream`.

The separate offline runtime-v3 artifact carries the exact P31 vLLM 0.30.0 native wheel, PyTorch/cu129 user-space closure, corrected TorchCodec 0.16.0+cu129, FlashInfer, manifests, and compact evidence. The Kaggle system CUDA compiler stayed at 12.8. A CUDA driver linker alias supports FlashInfer JIT without modifying the real driver.

P31 passed four-L4/SM89 inspection, NCCL world_size=4, native core and built-in FA2 import, and Gemma 4 31B QAT semantic chat at TP=4. The Gemma workload selected `TRITON_ATTN`. The new isolated SDK staging needs a clean Kaggle acceptance run using the existing corrected split-v2 Inputs and saved P31 Notebook Output; runtime-v3 stays an optional future artifact mode. No real-model FA2 workload has been claimed as passed, and none runs in the competition acceptance notebook.

Historical v0.2.0 remains a distinct T4x2/cu128 release.

This is a general-purpose offline inference/runtime SDK for compatible local models on the exact four-L4 profile. Gemma 4 is a separate acceptance workload, not a competition-specific dependency of the core SDK. Future v0.3.0 publication must add a new release and assets while preserving all existing v0.1.x/v0.2.0 tags, releases, assets, T4 native binaries, evidence, and model artifacts. GitHub assets should remain small; the 6+ GB runtime belongs in the separate Hugging Face binaries repository after fresh-session and license review.

Runtime-v3 archive SHA256: `0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73` (6,633,730,070 bytes; 215 members).
