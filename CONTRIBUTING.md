# Contributing

Thank you for improving `kaggle-vllm`. Keep the boundary clear: this repository
delivers and validates an upstream runtime; it does not reimplement vLLM.

## CPU development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests examples scripts
.venv/bin/python -m build
.venv/bin/python -m twine check dist/*
```

Do not install the native CUDA wheel globally. Tests and imports must not
download, activate or install anything implicitly. Do not add vLLM, Torch,
TorchVision, TorchAudio, CUDA or NCCL as regular SDK dependencies.

## Profiles and GPU evidence

Profile changes must cite wheel metadata/upstream requirements, update the
profile, overlay, dependency baseline, compatibility/provenance records and
tests together. A new Python ABI, GPU, CUDA family or topology needs a distinct
profile and a real Kaggle acceptance run.

For GPU reports include the Kaggle image/runtime date, Python, Torch/path,
Torch CUDA, toolkit, driver, driver-reported CUDA maximum, GPU names/SM, NCCL,
full argument-array command, complete relevant log and artifact SHA256. State
whether evidence is historical, newly executed, mocked or CPU-only.

## Repository hygiene and attribution

- Never commit native wheels, weights, model archives, caches, staged runtimes,
  generated logs or credentials.
- Preserve upstream vLLM tag/commit attribution; do not present upstream
  inference, tensor parallelism, sharding or serving as SDK inventions.
- Preserve model LICENSE and NOTICE terms. The Qwen artifact is not Apache-2.0
  and is not generally commercially licensed.
- Do not modify published tags or replace historical release/artifact bytes.

See [development](docs/development.md), [testing](docs/testing.md),
[security](docs/security.md) and [release process](docs/release.md).
