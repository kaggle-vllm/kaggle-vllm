# Python 3.13 SDK installation

`kaggle-vllm` 0.2.1 supports CPython 3.13 only. Its target is Kaggle Linux x86_64 with two Tesla T4 GPUs, SM75, CUDA 12.8, and TP=1 or TP=2. The SDK is pure Python. Its only normal dependency is `packaging>=23`; `vllm`, Torch, and CUDA are not SDK dependencies.

Install the published SDK with `python3.13 -m pip install "kaggle-vllm[hub]==0.2.1"`. To build from source, use `python3.13 -m build --no-isolation`, then install its `py3-none-any` wheel with `python3.13 -m pip install --no-deps --target <empty-test-dir> dist/kaggle_vllm-0.2.1-py3-none-any.whl`. No native runtime is installed by this step.

`kaggle-vllm bootstrap --dry-run --json` exposes the pending cp313 native artifact. A real bootstrap refuses to download or stage a wheel until its identity and dependency overlay have been validated and registered. The historical cp312 wheel cannot satisfy cp313 and the `stage-wheel` command checks ABI and platform tags before pip runs.

For environment inspection, run `kaggle-vllm fingerprint`, `kaggle-vllm doctor --json`, and `kaggle-vllm bootstrap --dry-run --json`. The Python API is illustrated in the [root README](../README.md). See [native rebuild preparation](python313-native-runtime-rebuild.md) for a separate native-runtime procedure.
