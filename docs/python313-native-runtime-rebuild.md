# Python 3.13 native runtime rebuild preparation

## Identity and target

- Keep upstream vLLM **v0.18.1**, commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`. Record every source patch if a CPython 3.13 build needs one. Do not silently substitute a later vLLM generation.
- Build on a real Kaggle Linux x86_64 session with **two Tesla T4 GPUs**, SM75, CUDA 12.8 family, CPython 3.13 (`cp313`), and the observed PyTorch 2.11.0+cu128 baseline. Confirm versions afresh before building.
- Keep `TORCH_CUDA_ARCH_LIST=7.5`, `VLLM_TARGET_DEVICE=cuda`, and the existing Torch installation. Historical scripts and the cp312 overlay in `compat/historical-py312-v0.2.0/` are evidence, not a validated Python 3.13 recipe.

## Compatibility risks to resolve

Historical vLLM 0.18.1 wheel metadata required `transformers>=4.56,<5`; the observed Kaggle image has Transformers 5.16.1. The historical protobuf lower bound was `>=5.29.6`; the observed image had 5.29.5. Determine whether a Python 3.13 compatible dependency overlay can satisfy vLLM 0.18.1 without replacing Kaggle Torch/CUDA, and document any source edits. These are known contract mismatches, not demonstrated build failures. Check Triton 3.6.0, Torch 2.11.0+cu128, CMake, compiler, CUDA toolkit, NCCL, and Python 3.13 headers in the actual build session.

## Build and verification record

1. Capture `python3.13 --version`, `python3.13 -m pip list`, Torch/CUDA/NCCL information, both GPU names and compute capability, compiler/CMake, toolkit, driver, source commit, patches, and build flags.
2. Build the native wheel from the pinned vLLM source in Kaggle. Confirm the wheel filename has real `cp313-cp313` and Linux x86_64 tags. Inspect its ZIP `WHEEL`/`METADATA` and compiled `.so` files; verify the CUDA extensions contain SM75 code. Do not rename or patch the cp312 wheel.
3. With a dependency plan documented separately, stage the wheel via `python3.13 -m pip install --no-deps --target <empty-dir> <new-wheel>`. Keep Kaggle's Torch and CUDA stack unchanged.
4. Run `import vllm`, `import vllm._C`, and `import vllm._moe_C`. Record versions and import paths. Run a raw two-rank NCCL smoke test, TP=1 inference, TP=2 inference, and the Qwen TP=2 `sharded_state` regression where applicable.
5. Confirm the selected attention backend on T4, preferably `TRITON_ATTN` if supported; run `/v1/models` and `/v1/chat/completions` checks against the OpenAI-compatible server.
6. Preserve logs and JSON evidence without rewriting historical 2026 Python 3.12 results. Compute `sha256sum <new-wheel>` and a checksum manifest for every new evidence file.
7. **Only after successful validation**, update `profile.json` with the actual wheel filename, SHA256, immutable Hub repository/revision if published, validated dependency overlay identity, and `functionally_validated` status. Update tests and run bootstrap dry-run before enabling actual download. Never invent these fields.

Current active profile fields `wheel.filename`, `wheel.sha256`, `wheel.hf_repo_id`, `wheel.hf_revision`, and `overlay.lock` are null. Native imports, NCCL, TP=1/TP=2, Qwen, attention backend, and serving remain unvalidated on CPython 3.13.
