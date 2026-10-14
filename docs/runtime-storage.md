# L4 runtime storage

The qualified Kaggle `/kaggle/working` volume had about 19.5 GiB writable capacity. The expanded v0.3 inference runtime belongs under `/tmp/kaggle-vllm/v0.3.0/l4x4`, with a conservative 24 GiB free-space guard before installation. If scratch space is too small, bootstrap reports available and required bytes and stops. It never falls back to `/kaggle/working`.

An advanced `--runtime-root` override must name an SDK-owned `kaggle-vllm-v030-runtime` directory, avoid symlink traversal, and stay outside `/kaggle/working` and `/kaggle/input`. Reset requires an existing matching SDK manifest plus `--reset-runtime --yes`; `--dry-run` performs no mutations.

`/kaggle/working` is for compact evidence, logs, manifests, and explicitly requested persistent output. The 168 dependency wheels in `L4_INFERENCE_LOCK.json` plus the one native wheel are the successful P33 selection. The 195-wheel runtime-v3 closure is archival and must not be treated as the default inference install.
