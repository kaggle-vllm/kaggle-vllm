# kaggle-vllm v0.2.1

Version 0.2.1 publishes the pure-Python SDK for **CPython 3.13 only**. Its
compatibility target is Kaggle Linux x86_64 with two NVIDIA Tesla T4 GPUs,
SM75, a CUDA 12.8 environment, and TP=1 or TP=2 interfaces.

Install the SDK with:

```bash
python3.13 -m pip install "kaggle-vllm[hub]==0.2.1"
```

The SDK offers `kaggle-vllm fingerprint`, `doctor --json`,
`bootstrap --dry-run --json`, native-wheel checksum and ABI checks, and the
`KaggleLLM` and sharded-state APIs. Its wheel and source distribution contain
no native vLLM runtime, PyTorch, CUDA libraries, model weights, or NVIDIA
binaries. Inference and serving require a separately installed compatible
upstream vLLM runtime.

Package metadata, versioned diagnostics, installation examples, and CPU tests
are aligned with Python 3.13. The release retains the v0.2.0 Python 3.12 GPU
acceptance, benchmark, and checksum records without treating them as Python
3.13 GPU validation. See the [README](https://github.com/kaggle-vllm/kaggle-vllm/blob/v0.2.1/README.md)
and [changelog](https://github.com/kaggle-vllm/kaggle-vllm/blob/v0.2.1/CHANGELOG.md).

Validation for this release: Python 3.13 CPU suite **95 passed, 1 skipped**;
static profile/provenance and Markdown-link checks passed; wheel and sdist
built successfully and passed strict Twine validation. GPU inference and
benchmark tests were not run for this version.
