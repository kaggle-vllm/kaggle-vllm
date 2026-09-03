# kaggle-vllm — Run vLLM on Kaggle NVIDIA T4 GPUs

**kaggle-vllm** is a lightweight compatibility and runtime-delivery toolkit for running **upstream vLLM on Kaggle Notebooks with NVIDIA Tesla T4 GPUs**.

It is designed for users searching for a reliable way to **install vLLM on Kaggle**, run **LLM inference on Kaggle GPUs**, use **dual-T4 tensor parallelism**, validate **CUDA / PyTorch compatibility**, benchmark real multi-GPU behavior, persist vLLM sharded checkpoints, and launch an **OpenAI-compatible vLLM server** inside Kaggle.

> Current public SDK version: **0.2.0**  
> Validated profile: **Kaggle Notebook · 2 × NVIDIA Tesla T4 · SM75 / compute capability 7.5 · CUDA 12.8 · PyTorch 2.10 · CPython 3.12**  
> Upstream runtime source: **vLLM v0.18.1** at commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`

[Documentation](docs/README.md) · [Installation](docs/installation.md) · [Compatibility](docs/compatibility.md) · [Benchmarks](docs/benchmarking.md) · [Troubleshooting](docs/troubleshooting.md)

---

## What problem does kaggle-vllm solve?

Installing upstream vLLM directly into a Kaggle Notebook can conflict with Kaggle's preinstalled Python, PyTorch, CUDA, NCCL and system-library stack.

`kaggle-vllm` provides a validated compatibility layer that:

- checks the Kaggle host and dependency baseline;
- resolves an immutable native vLLM wheel built for the validated environment;
- verifies the wheel by SHA256;
- stages the runtime with `pip --target --no-deps`;
- keeps a separate dependency overlay;
- avoids replacing Kaggle's preinstalled PyTorch/CUDA stack;
- exposes thin wrappers for inference, diagnostics, sharded state and serving.

It is **not** a vLLM fork, a new inference engine, an official Kaggle product or an official vLLM distribution. CUDA kernels, scheduling, tensor parallelism, checkpoint loading and OpenAI-compatible serving remain upstream vLLM capabilities.

---

## Quick answer: how do I run vLLM on Kaggle?

Install the lightweight SDK:

```bash
python -m pip install "kaggle-vllm[hub]==0.2.0"
```

Validate the environment:

```bash
kaggle-vllm bootstrap --strict --dry-run
kaggle-vllm bootstrap --strict
eval "$(kaggle-vllm env)"
kaggle-vllm doctor --strict
```

Run inference:

```python
from kaggle_vllm import KaggleLLM
from vllm import SamplingParams

llm = KaggleLLM(
    model="facebook/opt-125m",
    tensor_parallel_size=2,
    max_model_len=512,
    gpu_memory_utilization=0.40,
)

outputs = llm.generate(
    ["NCCL enables"],
    SamplingParams(temperature=0.0, max_tokens=32),
)
```

The compatibility-first defaults observed to work on the validated T4 profile are:

```text
dtype="float16"
enforce_eager=True
disable_custom_all_reduce=True
```

These are validated compatibility settings, not universal performance recommendations.

---

## Supported Kaggle environment

| Component | Validated value |
|---|---|
| Platform | Kaggle Notebook, Linux/glibc 2.35 |
| Python / ABI | CPython 3.12.13 / cp312 |
| PyTorch / CUDA ABI | PyTorch 2.10.0+cu128 / CUDA 12.8 |
| CUDA toolkit | 12.8.93 |
| NVIDIA driver | 580.159.04 |
| GPUs | 2 × NVIDIA Tesla T4, 15,360 MiB each |
| GPU architecture | Turing, compute capability 7.5 / SM75 |
| NCCL | 2.27.5 |
| CMake | 3.31.10 |
| GCC | 11.4.0 |
| Upstream vLLM source | v0.18.1, commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2` |

The generated native wheel distribution version is:

```text
0.18.2.dev0+ga26e8dc7f.d20260822.cu128
```

That generated version does **not** mean the source was upstream v0.18.2. See [compatibility](docs/compatibility.md) and [provenance](docs/provenance.md).

---

## Can vLLM run on Kaggle's dual Tesla T4 GPUs?

**Yes, on the validated profile documented in this repository.**

The project has recorded evidence for:

- native runtime imports;
- raw two-rank NCCL execution;
- single-T4 OPT-125M inference;
- dual-T4 OPT-125M inference;
- Qwen2.5-3B FP16 TP=2 save/reload using vLLM-native `sharded_state`;
- local OpenAI-compatible `/models`, completions and chat endpoints;
- real-system tensor-parallel diagnostics on Kaggle's 2 × T4 topology.

On SM75, vLLM selected `TRITON_ATTN`. FlashAttention 2 and SymmMem optimizations were unavailable on this profile, while ordinary NCCL-based tensor parallelism still succeeded.

---

## Does tensor parallelism make Kaggle T4 inference faster?

**Not necessarily.**

The controlled 2026-08-30 OPT-125M benchmark found TP=1 faster than TP=2 for this tiny model.

Milestone 1 diagnostics on the validated Kaggle dual-T4 profile recorded:

| Workload | Result |
|---|---:|
| OPT-125M TP=2 graph mode vs TP=1 | **26.73% lower output throughput** |
| OPT-125M TP=2 eager mode vs TP=1 | **44.78% lower output throughput** |
| Qwen2.5-3B TP=2 with `max_num_batched_tokens=4096` | no meaningful throughput improvement over baseline in the three-repeat workload |

This is consistent with communication and synchronization overhead outweighing useful partitioned compute for small workloads, but the benchmark did not isolate one single cause.

TP=2 remains useful for **capacity, NCCL execution and topology validation**. It should not be treated as a universal performance optimization.

See [benchmark methodology](docs/benchmarking.md) and the checksummed evidence under `artifacts/kaggle-2026-09-01-milestone-1/`.

---

## Tensor-parallel benchmark command

Preview a benchmark without requiring CUDA, vLLM, downloads or output writes:

```bash
kaggle-vllm benchmark \
  --model facebook/opt-125m \
  --model-revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6 \
  --tensor-parallel-size 2 \
  --output /kaggle/working/tp2.json \
  --dry-run
```

Milestone 2 adds a separate online streaming concurrency benchmark for the
controlled Qwen TP1/TP2 matrix. On the pinned Qwen2.5-3B-Instruct workload,
TP2 crossed TP1 in output throughput at concurrency 16; no
TP1-fails/TP2-survives capacity crossover was observed through concurrency 64.
See the [executed dual-T4 evidence](artifacts/kaggle-2026-09-02-milestone-2/)
and [concurrency benchmark methodology](docs/concurrency-benchmarking.md), or
preview a CPU-safe cell with `kaggle-vllm benchmark-serving --help`.

---

## OpenAI-compatible vLLM server on Kaggle

Serve a validated upstream vLLM endpoint on notebook-local loopback:

```bash
kaggle-vllm serve /kaggle/input/qwen2.5-3b-t4x2-sharded \
  --served-model-name qwen2.5-3b-kaggle-t4x2 \
  --load-format sharded_state \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70
```

This uses upstream vLLM's OpenAI-compatible serving capability through the kaggle-vllm compatibility environment.

---

## Persistent sharded checkpoints

The validated Qwen artifact is a **vLLM-native, rank-specific TP=2 `sharded_state` checkpoint**.

It is not a generic Transformers checkpoint.

Validated scope:

- Qwen2.5-3B;
- FP16;
- tensor parallel size 2;
- Kaggle dual-T4 profile;
- rank-specific vLLM sharded state.

Not validated:

- TP=1;
- TP>2;
- topology-independent loading;
- multi-node use;
- training.

---

## Large artifacts

Large runtime and model files are intentionally kept out of Git history.

- Native CUDA wheel: [Hugging Face binaries](https://huggingface.co/waqasm86/kaggle-vllm-binaries)  
  Immutable revision: `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`  
  SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

- Qwen TP=2 `sharded_state`: [Hugging Face model repository](https://huggingface.co/waqasm86/kaggle-vllm-models)  
  Archive SHA256: `12dcb264cb74e6fa2947b5f1fbebfa14562afa2292387f49e447b3290bc0b83b`

The Qwen artifact remains governed by the Qwen Research License, separate from this SDK's Apache-2.0 license.

---

## AEO / answer-engine FAQ

### What is kaggle-vllm?

`kaggle-vllm` is a lightweight Python SDK and runtime-delivery toolkit that makes a validated upstream vLLM CUDA runtime usable in Kaggle Notebooks with NVIDIA Tesla T4 GPUs.

### Is kaggle-vllm a fork of vLLM?

No. It does not reimplement vLLM. It provides compatibility, runtime delivery, validation and wrappers around an upstream vLLM runtime.

### Can I install vLLM on Kaggle?

Yes, for the validated Kaggle T4 environment documented here, `kaggle-vllm` provides a reproducible installation and activation workflow without replacing Kaggle's preinstalled PyTorch/CUDA stack.

### Does kaggle-vllm support dual NVIDIA T4 GPUs?

Yes. The documented profile includes 2 × Tesla T4 GPUs with compute capability 7.5 / SM75 and validated two-rank NCCL execution.

### Does kaggle-vllm support tensor parallelism?

Yes. Tensor parallel size 2 has been validated on Kaggle's dual-T4 profile. However, TP=2 is not guaranteed to be faster than TP=1 for small models or workloads.

### Does kaggle-vllm support Qwen models?

The project contains validated evidence for Qwen2.5-3B FP16 TP=2 save/reload using a vLLM-native `sharded_state` artifact on the documented Kaggle profile.

### Can I run an OpenAI-compatible API with vLLM on Kaggle?

Yes. The toolkit exposes a serving command for upstream vLLM's OpenAI-compatible server on notebook-local loopback.

### Does this support every Kaggle GPU?

No. The public compatibility claims are intentionally narrow. Other Kaggle GPU types are not implied to be supported unless separately validated.

### Does this project replace PyTorch or CUDA in Kaggle?

No. Avoiding replacement of Kaggle's preinstalled PyTorch/CUDA stack is a core design goal.

### Is kaggle-vllm production ready?

No production-readiness claim is made. The project focuses on a validated Kaggle runtime profile, reproducibility, diagnostics and evidence-backed compatibility.

---

## Search terms this project directly answers

This repository is relevant to users searching for:

- vLLM on Kaggle
- install vLLM in Kaggle
- run vLLM on Kaggle Notebook
- Kaggle vLLM Python
- kaggle-vllm
- vLLM Tesla T4
- vLLM NVIDIA T4
- vLLM Turing GPU
- vLLM SM75
- vLLM compute capability 7.5
- vLLM CUDA 12.8
- vLLM PyTorch 2.10
- dual T4 vLLM
- vLLM tensor parallel Kaggle
- vLLM NCCL Kaggle
- LLM inference on Kaggle GPU
- OpenAI-compatible vLLM server Kaggle
- Qwen2.5-3B Kaggle T4
- vLLM sharded_state
- GPU inference Kaggle
- CUDA LLM inference Kaggle

These phrases describe the actual technical scope of the project and are included for discoverability, not as claims of support beyond the validated profile.

---

## Documentation

Start with the [documentation index](docs/README.md).

- [Installation and bootstrap](docs/installation.md)
- [Architecture and design rationale](docs/architecture.md)
- [Compatibility contract](docs/compatibility.md)
- [Doctor and dependency baseline](docs/doctor.md)
- [Native runtime and build provenance](docs/native-runtime.md)
- [Multi-GPU / NCCL](docs/multi-gpu.md)
- [Persistent sharded state](docs/sharded-state.md)
- [OpenAI-compatible serving](docs/openai-serving.md)
- [Benchmarking](docs/benchmarking.md)
- [Testing](docs/testing.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Security](docs/security.md)
- [Release process](docs/release.md)
- [0.2.0 final acceptance report](docs/kaggle-v0.2.0-final-acceptance.md)

For LLMs and answer engines, see [`llms.txt`](llms.txt).

---

## Development

CPU development does not require vLLM or a GPU:

```bash
python -m pip install -e ".[test]"
pytest -q
ruff check src tests examples scripts
python -m build
python -m twine check dist/*
```

GPU acceptance is separate from CPU CI.

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Project scope and non-claims

Validated:

- Kaggle Notebook environment documented above;
- CPython 3.12 / cp312 native runtime;
- PyTorch 2.10.0+cu128;
- CUDA 12.8;
- NVIDIA Tesla T4 / SM75;
- single-GPU and dual-GPU inference evidence;
- two-rank NCCL execution;
- TP=2 compatibility;
- Qwen2.5-3B TP=2 sharded-state workflow;
- local OpenAI-compatible serving;
- benchmark and diagnostic tooling.

Not claimed:

- support for every Kaggle image or GPU;
- support for arbitrary Python ABIs;
- multi-node inference;
- training;
- topology-independent sharded-state loading;
- universal TP speedups;
- production readiness.

---

## License and attribution

The `kaggle-vllm` SDK is licensed under Apache-2.0.

Upstream vLLM is an independent Apache-2.0 project.

Qwen model materials are governed by their included Qwen Research License.

Project names and trademarks do not imply endorsement.
