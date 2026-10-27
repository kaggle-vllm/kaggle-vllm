# Documentation

The [root README](../README.md) describes the 0.2.1 Python 3.13 SDK for Kaggle dual T4. The active profile is `src/kaggle_vllm/profiles/kaggle-t4x2-cu128/profile.json`. [Native rebuild preparation](python313-native-runtime-rebuild.md) records a separate Kaggle runtime procedure.

[Installation](installation.md), [compatibility](compatibility.md), [doctor](doctor.md), [native runtime](native-runtime.md), [Kaggle runtime](kaggle-runtime.md), and [building vLLM](building-vllm.md) describe the active pending target. Other older documents in this directory and the original README snapshot in [HISTORICAL-README-PY312.md](../HISTORICAL-README-PY312.md) describe the **historical Python 3.12/cp312 validation and release workflow**. Their successful native imports, NCCL, TP=2, Qwen, and serving results are not Python 3.13 validation. The original profile and overlay are preserved in `compat/historical-py312-v0.2.0/`; executed evidence remains in `artifacts/`.
