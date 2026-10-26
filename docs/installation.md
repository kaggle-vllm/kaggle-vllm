# Local Python 3.13 SDK installation

This source tree is `kaggle-vllm` 0.2.0 for CPython 3.13, Kaggle dual Tesla T4, SM75, and CUDA 12.8. The SDK is pure Python. It has only `packaging>=23` as a normal dependency; `vllm`, Torch, and CUDA are not SDK dependencies.

Build the SDK locally with `python3.13 -m build`, then install its `py3-none-any` wheel with `python3.13 -m pip install --no-deps --target <empty-test-dir> dist/kaggle_vllm-0.2.0-py3-none-any.whl`. No native runtime is installed by this step.

`kaggle-vllm bootstrap --dry-run --json` exposes the pending cp313 native artifact. A real bootstrap refuses to download or stage a wheel until its identity and dependency overlay have been validated and registered. The historical cp312 wheel cannot satisfy cp313 and the `stage-wheel` command checks ABI and platform tags before pip runs.

The public `kaggle-vllm==0.2.0` package predates this local Python 3.13 reconstruction. Do not assume `pip install kaggle-vllm==0.2.0` retrieves these local changes. See [native rebuild preparation](python313-native-runtime-rebuild.md).
