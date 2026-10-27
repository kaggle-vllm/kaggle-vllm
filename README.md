# kaggle-vllm 0.2.1 — Python 3.13 SDK for Kaggle dual T4

`kaggle-vllm` is a pure-Python compatibility and runtime-delivery SDK for
Kaggle Linux x86_64 with **two NVIDIA Tesla T4 GPUs** (compute capability 7.5 /
SM75), a CUDA 12.8 environment, and tensor parallel sizes 1 or 2. Version
0.2.1 supports **CPython 3.13 only** (`>=3.13,<3.14`).

The SDK provides environment checks, dependency diagnostics, native-wheel
checksum and ABI validation, explicit runtime staging, a lazy `KaggleLLM`
wrapper, sharded-state inspection, and an OpenAI-compatible server command.
The Python distribution contains no native vLLM runtime, CUDA libraries,
PyTorch, model weights, or NVIDIA binaries. Inference and serving require a
separately installed compatible upstream vLLM runtime.

## Install

```bash
python3.13 -m pip install "kaggle-vllm[hub]==0.2.1"
```

The normal SDK dependency is `packaging>=23`; the optional `hub` extra adds
`huggingface_hub`. The package does not install or replace Kaggle's PyTorch or
CUDA stack.

## CLI and environment checks

```bash
kaggle-vllm --help
kaggle-vllm fingerprint
kaggle-vllm doctor --json
kaggle-vllm bootstrap --dry-run --json
kaggle-vllm verify-gpus --tensor-parallel-size 2
```

`fingerprint` reports the current host. `doctor` compares it with the packaged
Kaggle T4 profile and returns a failure status when a required condition is
unmet. `bootstrap --dry-run` is read-only; actual staging requires a validated
runtime identity in the profile. `verify-wheel` and `stage-wheel` check a native
wheel before staging it. See the [CLI and installation guide](docs/installation.md).

## Python API

With a compatible upstream vLLM runtime available in the process environment:

```python
from kaggle_vllm import KaggleLLM, inspect_sharded_model, load_profile
from vllm import SamplingParams

profile = load_profile()
print(profile.name, profile.python_abi)

llm = KaggleLLM(
    model="facebook/opt-125m",
    tensor_parallel_size=2,
    max_model_len=512,
    gpu_memory_utilization=0.40,
)
outputs = llm.generate(
    ["Hello from Kaggle"], SamplingParams(temperature=0.0, max_tokens=32)
)

# For an existing vLLM sharded_state directory:
# inspection = inspect_sharded_model("/kaggle/input/my-sharded-model")
```

The API also accepts `tensor_parallel_size=1`. Importing `kaggle_vllm` alone
does not initialize a GPU, access the network, or install packages.

## Compatibility and provenance

The active package profile is
[`kaggle-t4x2-cu128`](src/kaggle_vllm/profiles/kaggle-t4x2-cu128/profile.json).
It records the Python 3.13, dual-T4, SM75 and CUDA 12.8 target and preserves
strict wheel ABI validation. The [compatibility guide](docs/compatibility.md)
distinguishes this SDK target from the historical Python 3.12 GPU results.

The original v0.2.0 acceptance, benchmark, native-wheel checksums, and model
provenance remain in [`artifacts/`](artifacts/),
[`compat/historical-py312-v0.2.0/`](compat/historical-py312-v0.2.0/), and
[`release/`](release/). Those records do **not** establish Python 3.13 native
imports, NCCL, inference, serving, or benchmark performance. See the
[documentation index](docs/README.md) and [changelog](CHANGELOG.md).

This project is not an official Kaggle or upstream vLLM distribution. The SDK
is Apache-2.0; upstream vLLM and model materials retain their own licenses.
