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

> **Status:** The lightweight SDK release is v0.1.2 and is published on PyPI.
> The native runtime remains a separate, explicit Hugging Face bootstrap.
> On 2026-08-25, v0.1.1 completed the full post-publication delivery/Qwen
> acceptance, and v0.1.2 completed a focused safe-reset, re-bootstrap, and
> TP=2 acceptance run on the documented Kaggle dual-T4 environment. These are
> configuration-specific validation results, not production-readiness claims.

## Validated environment

The original 2026-08-22/23 evidence and the fresh 2026-08-25 acceptance run
recorded:

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

The SDK is published on [PyPI](https://pypi.org/project/kaggle-vllm/) and has
no hard dependency on vLLM, Torch, or CUDA. The recommended Kaggle flow
installs Hub support and enforces the validated runtime profile:

```bash
pip install "kaggle-vllm[hub]==0.1.2"
kaggle-vllm bootstrap --strict
```

The underscore spelling is normalized to the same PyPI distribution:

```bash
pip install "kaggle_vllm[hub]==0.1.2"
```

The canonical distribution spelling is equivalent:

```bash
python -m pip install kaggle-vllm
```

`pip install kaggle_vllm` installs only the small `kaggle-vllm` distribution;
Python packaging normalizes `_` and `-` in project names. The explicit
`bootstrap` command then downloads the exact native wheel from the Hugging Face
Hub/Xet-backed repository, checks its immutable revision and SHA256, stages it
with `pip --target --no-deps`, and creates the validated dependency overlay.
It never replaces or reinstalls Kaggle's Torch packages.

Importing `kaggle_vllm` never downloads or installs anything. The native wheel
is CPython 3.12 (`cp312`) and bootstrap rejects Python 3.11 even though the
lightweight SDK itself can be developed and tested with Python 3.11. An
immutable Hugging Face SDK fallback is documented in
[installation](docs/installation.md).

Inspect the complete plan without network or filesystem changes:

```bash
kaggle-vllm bootstrap --dry-run --strict
```

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

## Explicit native bootstrap and activation

Normal dependency resolution can replace Kaggle's tightly coupled Torch/CUDA
packages. Bootstrap uses the packaged `kaggle-t4x2-cu128` profile and pins the
native artifact to Hugging Face commit
`f6b4f10de54924ed6fe9e28cceab84eca7276ab6`:

```bash
kaggle-vllm bootstrap --strict
eval "$(kaggle-vllm env)"  # optional for subsequent shell commands
```

By default it uses `/kaggle/working/vllm-staged`,
`/kaggle/working/vllm-runtime-overlay`, and
`/kaggle/working/kaggle-vllm-cache`; every path is overridable. The packaged
overlay lock is the exact small reproducibility input from the successful
Kaggle recovery. Bootstrap rejects `torch`, `torchvision`, and `torchaudio`
entries, writes a runtime manifest, and refuses incompatible non-empty runtime
directories. `KaggleLLM` may activate an already-completed default manifest,
but it never bootstraps implicitly. See [installation](docs/installation.md).

### Recovering an owned staged runtime

Bootstrap still refuses to overwrite non-empty destinations by default. From
v0.1.2, inspect a manifest-aware reset plan without changing anything:

```bash
kaggle-vllm bootstrap --reset-runtime --dry-run --strict
```

After reviewing the exact staged, overlay, manifest, and preserved-cache paths,
explicitly confirm the reset and continue bootstrap in the same invocation:

```bash
kaggle-vllm bootstrap --reset-runtime --yes --strict
```

Reset accepts only known default paths or custom paths proven by the selected
runtime manifest. It rejects root/system/Kaggle parent paths, the current
repository, path overlap, and symlink traversal. The download cache is
preserved by default. See [installation recovery](docs/installation.md#recovering-from-an-existing-staged-runtime)
and [issue #7](https://github.com/kaggle-vllm/kaggle-vllm/issues/7).

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

The [executed v0.1.1 acceptance notebook](kaggle-notebooks/kaggle_vllm_0_1_1_dual_t4_acceptance.ipynb)
and its [evidence summary](docs/kaggle-v0.1.1-acceptance.md) freshly validated:

- PyPI `kaggle-vllm==0.1.1` installation without replacing PyTorch
- strict `kaggle-t4x2-cu128` compatibility checks
- canonical Hugging Face binary delivery at the immutable revision and SHA256
- successful staging and imports of `vllm`, `vllm._C`, `vllm._moe_C`, and
  `vllm.cumem_allocator`
- real `facebook/opt-125m` inference with TP=2 over both Tesla T4 GPUs
- canonical Qwen repository download and valid inspection of two ranks/four
  rank-specific shards
- Qwen TP=2 `sharded_state` reload and successful text generation

The focused [v0.1.2 reset acceptance notebook](kaggle-notebooks/kaggle_vllm_0_1_2_reset_acceptance.ipynb)
and [evidence summary](docs/kaggle-v0.1.2-reset-acceptance.md) additionally
validated the manifest-aware reset workflow: the default non-empty refusal,
non-mutating reset dry-run, explicit confirmed reset, cache preservation,
strict re-bootstrap, staged native imports, preserved Kaggle Torch, and real
OPT-125M NCCL TP=2 generation. It intentionally did not repeat the multi-GB
Qwen download because the persistent model and native runtime were unchanged.

The broader original validation also covered:

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
kaggle-vllm bootstrap [--strict] [--dry-run] [--reset-runtime [--yes]]
kaggle-vllm env [--manifest PATH]
kaggle-vllm verify-gpus --tensor-parallel-size 2
kaggle-vllm inspect-shards PATH --json
kaggle-vllm verify-wheel PATH [--sha256 DIGEST]
kaggle-vllm stage-wheel PATH --target TARGET [--sha256 DIGEST]
kaggle-vllm serve MODEL ...
```

## Artifact distribution and security

Verified release artifacts are published separately from the source repository:

- [validated Kaggle dual-T4 vLLM wheel and metadata](https://huggingface.co/waqasm86/kaggle-vllm-binaries)
- [Qwen2.5-3B-Instruct TP=2 persistent sharded state](https://huggingface.co/waqasm86/kaggle-vllm-models)

Large wheels, archives, safetensors, caches, overlays, and extracted models are
ignored by Git. Published artifacts must carry checksums, compatibility data,
and upstream attribution. Never commit Kaggle, GitHub, or Hugging Face tokens.
The Qwen persistent checkpoint remains governed by the non-commercial Qwen
Research License included with the model, not this repository's Apache-2.0
license.

## Known limitations

- Validation is specific to the tabled Kaggle environment and CPython 3.12 ABI.
- The SDK supports Python 3.10+, but the published native wheel profile is
  Linux x86_64 CPython 3.12 only.
- No local GPU test is claimed; GPU results come from executed Kaggle notebooks
  and the curated Kaggle evidence.
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
