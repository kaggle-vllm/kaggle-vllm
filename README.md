# kaggle-vllm

`kaggle-vllm` is a lightweight Python SDK and compatibility toolkit around
[upstream vLLM](https://github.com/vllm-project/vllm) for Kaggle's NVIDIA Tesla
T4 environment. It validates the runtime, protects Kaggle's preinstalled
PyTorch/CUDA stack during explicit artifact staging, wraps `vllm.LLM`, inspects
vLLM-native persistent sharded checkpoints, and safely launches vLLM's
OpenAI-compatible server.

It is **not a fork, reimplementation, or replacement for vLLM**. Inference,
tensor parallelism, sharded-state persistence, and serving remain upstream vLLM
capabilities.

> **Status:** v0.1 release candidate. Functionally validated on the documented
> Kaggle dual-T4 environment. It is not described as production-ready.

## Validated environment

The archived 2026-08-22/23 Kaggle runs recorded:

| Component | Validated value |
|---|---|
| Platform | Kaggle Notebook, Linux/glibc 2.35 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 (preserved system install) |
| CUDA toolkit | 12.8.93 |
| Driver | 580.159.04; `nvidia-smi` CUDA capability 13.0 |
| GPU | 2 × NVIDIA Tesla T4, 15,360 MiB each |
| Compute capability | 7.5 / SM75 |
| NCCL | 2.27.5 |
| CMake / GCC | 3.31.10 / 11.4.0 |
| vLLM source | tag v0.18.1, commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2` |

The generated wheel is:

```text
vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl
SHA256 5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c
```

The source identity and distribution version are not contradictory. The source
checkout is upstream `v0.18.1` at the commit above; the wheel filename/version
is generated build metadata from vLLM's `setuptools_scm` configuration, which
reported the next development version plus Git/date and local CUDA metadata.
It does not mean the source was the upstream v0.18.2 release.

## Install the lightweight SDK

The SDK intentionally has no hard dependency on vLLM or Torch:

```bash
python3 -m pip install -e .
kaggle-vllm fingerprint
kaggle-vllm doctor
```

Importing diagnostics never installs packages and does not require vLLM. The
validated binary is CPython 3.12 (`cp312`) and is distributed separately; it is
not part of the SDK source distribution.

## Python inference API

```python
from kaggle_vllm import KaggleLLM

llm = KaggleLLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    tensor_parallel_size=2,
    max_model_len=2048,
    gpu_memory_utilization=0.70,
)
outputs = llm.generate(["Explain tensor parallelism."], sampling_params)
```

`KaggleLLM` lazily imports and wraps upstream `vllm.LLM`. It validates the TP
degree against visible GPUs and forwards advanced keyword arguments. The SDK
supplies the conservative settings validated on Kaggle T4 by default:

```python
dtype="float16"
enforce_eager=True
disable_custom_all_reduce=True
```

These are validated conservative defaults for this Kaggle T4 configuration,
not claims of universal optimality. Every value can be overridden explicitly.

## Tensor parallelism is not persistent sharding

Runtime tensor parallelism partitions model execution across visible devices:

```python
KaggleLLM(model="Qwen/Qwen2.5-3B-Instruct", tensor_parallel_size=2)
```

A persistent TP-aware checkpoint is a different artifact. The experiment used
vLLM's native `save_sharded_state` machinery to write rank-specific files:

```text
model-rank-0-part-0.safetensors
model-rank-0-part-1.safetensors
model-rank-1-part-0.safetensors
model-rank-1-part-1.safetensors
```

Save and inspect one through the wrapper:

```python
inspection = llm.save_sharded_model(
    "/kaggle/working/qwen2.5-3b-t4x2-sharded"
)
print(inspection.rank_count)  # 2
```

Reload using the topology for which it was created:

```python
llm = KaggleLLM(
    model="/kaggle/input/qwen2.5-3b-t4x2-sharded",
    tensor_parallel_size=2,
    load_format="sharded_state",
    max_model_len=2048,
    gpu_memory_utilization=0.70,
)
```

This is not arbitrary tensor splitting, uneven 1/3–2/3 GPU allocation, or a
claim of topology-independent portability. See [persistent sharded state](docs/sharded-state.md).

## Explicit wheel staging and dependency overlay

Normal dependency resolution can replace Kaggle's tightly coupled Torch/CUDA
packages. The validated recovery staged the wheel without dependencies, then
placed a pinned dependency overlay ahead of it on `PYTHONPATH` while retaining
Kaggle's system Torch:

```bash
kaggle-vllm verify-wheel /path/to/vllm-*.whl --sha256 5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c
kaggle-vllm stage-wheel /path/to/vllm-*.whl --target /kaggle/working/vllm-staged
```

The Python installation helpers reject Torch entries in overlay requirements.
All installation operations are explicit; importing `kaggle_vllm` never
modifies global packages. See [installation](docs/installation.md).

## Kaggle CUDA-driver discovery

The toolkit was at `/usr/local/cuda-12.8`, while the mounted live driver was
`/usr/local/nvidia/lib64/libcuda.so`. CMake found the toolkit but initially did
not expose `CUDA::cuda_driver`. The successful build made the driver directory
visible with:

```bash
export CMAKE_LIBRARY_PATH=/usr/local/nvidia/lib64
```

The source-build scripts retain this workaround. The wheel itself is excluded
from Git.

## OpenAI-compatible serving

The server helper creates an argument array and invokes upstream `vllm serve`
without a shell:

```bash
kaggle-vllm serve /kaggle/input/qwen2.5-3b-t4x2-sharded \
  --served-model-name qwen2.5-3b-kaggle-t4x2 \
  --load-format sharded_state \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --host 127.0.0.1 \
  --port 8001
```

The archived Qwen run returned HTTP 200 from both `GET /v1/models` and
`POST /v1/chat/completions`. See [OpenAI serving](docs/openai-serving.md).

## What was functionally validated

- CUDA-enabled vLLM wheel build and SHA256 verification
- staged native imports (`vllm._C`, `vllm._moe_C`, allocator)
- isolated dependency overlay while preserving system PyTorch
- single-T4 FP16 inference with `facebook/opt-125m`
- raw two-rank NCCL all-reduce (`3.0` on both ranks)
- real vLLM TP=2 inference with `facebook/opt-125m`
- Qwen/Qwen2.5-3B-Instruct FP16 TP=2 inference
- persistent TP=2 sharded-state creation and reload
- OpenAI-compatible TP=2 serving from the sharded Qwen checkpoint

The curated evidence is in [`artifacts/kaggle-2026-08-23`](artifacts/kaggle-2026-08-23/README.md),
with the larger immutable evidence and model archives kept outside Git.

## Tesla T4 / SM75 behavior

FlashAttention 2 requires compute capability 8.0 or newer and was unavailable
on SM75. vLLM selected `TRITON_ATTN` in the recorded runs. SymmMem communicator
warnings are also expected because that capability is unavailable on SM75;
ordinary NCCL communication and TP=2 inference still completed successfully.

## CLI

```text
kaggle-vllm doctor
kaggle-vllm fingerprint
kaggle-vllm verify-gpus --tensor-parallel-size 2
kaggle-vllm inspect-shards PATH --json
kaggle-vllm verify-wheel PATH [--sha256 DIGEST]
kaggle-vllm stage-wheel PATH --target TARGET [--sha256 DIGEST]
kaggle-vllm serve MODEL ...
```

## Artifact distribution and security

Large wheels, archives, safetensors, caches, overlays, and extracted models are
ignored by Git. Published artifacts must carry checksums, compatibility data,
and upstream attribution. Never commit Kaggle, GitHub, or Hugging Face tokens.
The Qwen persistent checkpoint remains governed by the non-commercial Qwen
Research License included with the model, not this repository's Apache-2.0
license.

## Known limitations

- Validation is specific to the tabled Kaggle environment and CPython 3.12 ABI.
- No local GPU test is claimed; GPU results come from archived Kaggle evidence.
- The persistent model is TP-topology-aware and validated only at TP=2.
- The copied upstream HF weight index names original HF shards; standard
  Transformers loading is not supported. Use vLLM `sharded_state`.
- Eager execution/custom all-reduce settings were conservative correctness
  choices, not performance benchmarks.
- No arbitrary or uneven GPU-memory split API is provided.
- Qwen redistribution/use is non-commercial under its included license.

More detail: [architecture](docs/architecture.md), [runtime](docs/kaggle-runtime.md),
[tensor parallelism](docs/tensor-parallel.md), [validation](docs/validation.md),
and the [compatibility matrix](docs/compatibility-matrix.md).
