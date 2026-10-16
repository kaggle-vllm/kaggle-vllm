# Manual Kaggle acceptance

The two output-free notebooks in `kaggle-notebooks/` target the published 0.2.0 T4x2 and proposed 0.3.2 L4x4 releases separately. Both have Internet/DNS preflight, exact GPU checks, PyPI installation, strict bootstrap, native diagnostics, a matching TP generation case, cleanup observation and bounded JSON evidence. The L4 notebook downloads the public, pinned 23.3 GB Gemma 4 QAT model, checks configuration and processor, and repeats semantic TP4 prompts. It needs adequate scratch disk and a complete public runtime.

The notebooks passed JSON and Python syntax validation only. The user must upload and run them in fresh Kaggle sessions. P31 historical Gemma selected `TRITON_ATTN`; FA2 native import is distinct from FA2 model execution. Do not claim throughput or a 0.3.2 GPU PASS until the user's evidence files are reviewed.

## Publication order reminder

The 0.3.2 public notebook remains blocked until the native/metadata Hub namespace is complete at an immutable commit, all 168 publisher sources and the final SDK are verified, PR #39 is merged, and GitHub/PyPI publish the exact matching artifacts. Its preflight now checks PyPI, Hugging Face, PyTorch and FlashInfer. The separate 0.2.0 notebook targets the historic T4 runtime. Actual absolute paths and current statuses are in `21-manual-notebook-paths.md` and `.json`. Neither notebook was executed by Codex.
