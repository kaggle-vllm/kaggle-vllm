# Changelog

All notable changes are documented here.

## 0.2.0 — 2026-08-31

- Add dependency-aware, machine-readable runtime diagnostics sourced from the
  native wheel metadata and validated overlay.
- Correct the validated Tesla T4 / SM75 dependency profile so missing
  FlashInfer is reported as an optional, untested optimization rather than a
  core `TRITON_ATTN` compatibility failure.
- Add CPU CI across Python 3.10–3.13, package validation, provenance checks,
  documentation-link checks, and an explicit GPU/non-GPU testing boundary.
- Harden persistent sharded-state inspection against symlinks and topology
  mismatches.
- Record real 2026-08-30 dual-T4 development-candidate acceptance covering
  strict delivery and SHA256 verification, dependency doctor, native imports,
  preserved Kaggle system PyTorch, raw NCCL, OPT-125M TP=1/TP=2, local
  OpenAI-compatible HTTP endpoints and clean worker termination.
- Record the controlled 2026-08-30 TP benchmark. For the tiny OPT-125M
  workload, TP=1 was faster than TP=2; no universal TP=2 speedup is claimed.
- Record the successful 2026-08-31 fresh dual-T4 acceptance of the exact public
  `kaggle-vllm==0.2.0` package.
- Record the successful 2026-08-31 focused regression of the unchanged Qwen
  TP=2 persistent `sharded_state` artifact, including real load, generation,
  topology/symlink safety, and clean child-process exit.
- Retain the unchanged immutable native artifact built from upstream vLLM
  v0.18.1 at `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`; the SDK still does not
  install or replace Kaggle's Torch/CUDA stack as package dependencies.
- Reorganize user, operator, contributor, security, provenance, and release
  documentation and add final published-package acceptance preparation.

## 0.1.2 — 2026-08-25

- Add an explicit, manifest-aware `bootstrap --reset-runtime` recovery flow.
- Require `--yes` before removal, preserve the download cache, reject dangerous
  or symlinked targets, and support a structured non-mutating dry run.
- Keep the default bootstrap refusal for non-empty destinations unchanged.
- Attach the byte-identical PyPI 0.1.1 SDK wheel and sdist to the historical
  GitHub v0.1.1 release without duplicating native or model artifacts.

## 0.1.1 — 2026-08-25

- Rename the Hugging Face binary repository to `kaggle-vllm-binaries` and the
  persistent model repository to `kaggle-vllm-models`.
- Update the bootstrap profile and active documentation to the canonical
  repository IDs while preserving the immutable native revision and SHA256.
- No native vLLM wheel was rebuilt and no Qwen training, checkpoint
  regeneration, or large-artifact re-upload was performed.
- Record the successful post-publication Kaggle dual-T4 acceptance run covering
  strict bootstrap, staged native imports, OPT TP=2, and Qwen TP=2
  `sharded_state` generation.

## 0.1.0 — 2026-08-24

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
