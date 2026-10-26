# kaggle-vllm 0.2.0 — Python 3.13 dual-T4 development source

This local source tree is the lightweight `kaggle-vllm` 0.2.0 SDK for Kaggle Linux x86_64 with **exactly two NVIDIA Tesla T4 GPUs**, SM75, CUDA 12.8, and TP=1 or TP=2. Its Python package requires CPython 3.13. The SDK is pure Python and does not depend on `vllm`, `torch`, or CUDA packages at installation time.

**Status: target_unvalidated.** The Python 3.13 SDK is prepared for a future Kaggle native build. No validated cp313 vLLM CUDA wheel, checksum, Hub revision, or dependency overlay has been registered. `kaggle-vllm bootstrap --dry-run` reports this pending state; a real bootstrap stops before download or installation. `doctor` cannot report native compatibility as passed yet.

The historical 0.2.0 acceptance used CPython 3.12.13/cp312, PyTorch 2.10.0+cu128, upstream vLLM v0.18.1 at commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`, and dual T4 GPUs. Its wheel `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl` (SHA256 `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`) is **historical cp312 evidence, not a Python 3.13 runtime**. Acceptance logs, Qwen TP=2 regression, and benchmark data in `artifacts/` retain their original meaning; they do not prove Python 3.13 native compatibility.

The observed current Kaggle target has Python 3.13.15, PyTorch 2.11.0+cu128, Triton 3.6.0, Transformers 5.16.1, and two Tesla T4 GPUs. The historical vLLM dependency contract includes `transformers>=4.56,<5` and `protobuf>=5.29.6`; the observed Transformers 5.16.1 and protobuf 5.29.5 require investigation before a native runtime can be claimed. No upstream vLLM upgrade is assumed.

## Local SDK checks

```bash
python3.13 --version
PYTHONPATH=src python3.13 -c "import kaggle_vllm; print(kaggle_vllm.__version__)"
PYTHONPATH=src python3.13 -m kaggle_vllm.cli --help
PYTHONPATH=src python3.13 -m kaggle_vllm.cli bootstrap --dry-run --json
python3.13 -m build
```

The SDK build remains a pure Python `py3-none-any` wheel. The separate native vLLM wheel must be built with a real cp313 ABI on Kaggle. See [Python 3.13 native rebuild preparation](docs/python313-native-runtime-rebuild.md) and the [documentation index](docs/README.md).

The already published public `kaggle-vllm==0.2.0` files and GitHub v0.2.0 release are historical Python 3.12 artifacts. This Python 3.13 development source is prepared on branch `python313-t4-v020`; its locally rebuilt SDK files differ from the published files. Do not install the published cp312 native wheel into Python 3.13 or publish these development files under the existing release version. A future cp313 native build and real Kaggle dual-T4 validation are required before runtime compatibility can be claimed.

The SDK is Apache-2.0. Upstream vLLM and Qwen model materials retain their own licenses.
