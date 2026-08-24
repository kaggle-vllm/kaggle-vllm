# Changelog

All notable changes are documented here.

## 0.1.0 — publication pending

- Add the explicit `kaggle-vllm bootstrap` and side-effect-free dry run.
- Package the validated overlay lock and immutable `kaggle-t4x2-cu128` profile.
- Download the native wheel through Hugging Face Hub/Xet or a secure HTTPS
  fallback, pinned to an immutable revision and verified by SHA256.
- Reject non-cp312/platform-incompatible native bootstrap and optionally enforce
  the complete Kaggle dual-T4 profile with `--strict`.
- Record staged paths and provide explicit process/shell activation without
  modifying shell startup files.
- Keep the PyPI SDK dependency-free; vLLM, Torch, and CUDA remain outside
  `Requires-Dist`.

## 0.1.0-rc1 — 2026-08-24

- Add the lazy `KaggleLLM` wrapper around upstream `vllm.LLM`.
- Add Kaggle/T4/SM75 runtime diagnostics and TP-size validation.
- Add persistent `sharded_state` save delegation and structural inspection.
- Add streaming SHA256 verification and explicit `pip --target --no-deps` staging.
- Add subprocess-safe OpenAI-compatible server command generation.
- Add CPU-only unit coverage and a separately skipped Kaggle GPU profile test.
- Curate the 2026-08-23 Kaggle validation evidence.
- Document dual-T4 NCCL, TP=2, Qwen2.5-3B, sharded reload, serving, and limitations.

This release candidate is functionally validated only on the documented Kaggle
dual-T4 environment.
