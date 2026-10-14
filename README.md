# kaggle-vllm 0.3.0

A lightweight Python SDK for one Kaggle Notebook runtime: **Linux x86_64, CPython 3.12, exactly four NVIDIA L4 GPUs (SM89), tensor parallel size 4**. The SDK is pure Python and can be developed with Python 3.11. Its separate P31 native vLLM 0.30.0 wheel is cp312 only.

The historical T4 profile remains packaged. The accepted 0.3.0 SDK candidate wheel passed the cleanup-r1 four-L4 Kaggle run; this feature-branch source has changed and its rebuilt wheel requires another Kaggle run before release.

| SDK generation | Kaggle GPUs | SM | TP | Upstream vLLM | Status |
| --- | --- | ---: | ---: | --- | --- |
| 0.2.x | 2 × Tesla T4 | 75 | 2 | 0.18.1 | Historical validated release |
| 0.3.0 | Exactly 4 × NVIDIA L4 | 89 | 4 | 0.30.0 | Accepted candidate: 19/19 Kaggle gates; rebuilt wheel retest required |

## Runtime identity

The validated P31 native wheel was built from vLLM v0.30.0 at commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`. Its SHA256 is `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`. The canonical P31 final-v2 tar SHA256 is `e82085d0c6f180d2f4f8eb8f52332a92051273715e0923082fc65fc8857ddad5`.

The runtime uses PyTorch `2.13.0+cu129`, torchvision `0.28.0+cu129`, torchaudio `2.11.0+cu129`, Triton `3.7.1`, TorchCodec `0.16.0+cu129`, and FlashInfer `0.6.18.post1`. Kaggle's system CUDA compiler remains 12.8; cu129 is a PyTorch user-space closure. No system CUDA 12.9 toolkit is installed by this SDK.

P31 qualified vLLM's built-in FlashAttention-2 extension as compiled, packaged, SM89 targeted, and importable. Backend choice is model-specific. The validated Gemma 4 31B QAT workload selected `TRITON_ATTN`; no Gemma FA2 claim is made. P31 Step 20 passed semantic chat generation with `AutoProcessor` and TP=4. The earlier raw Step 17 newline-like output did not establish semantic success.

## Kaggle use

For a fresh run of the rebuilt wheel, attach the small SDK package, corrected split-v2 wheelhouse Datasets, and the saved P31 Notebook Output. The [final SDK acceptance notebook](kaggle-notebooks/kaggle_vllm_0_3_0_l4x4_final_sdk_acceptance.ipynb) uses the exact P33 lean lock: 168 dependency wheels plus the native wheel. The broader 195-wheel runtime-v3 manifest remains archival. Bootstrap installs under `/tmp/kaggle-vllm/v0.3.0/l4x4/` after a free-space check, with pip `--no-index --no-deps --no-cache-dir --no-compile`. It does not resolve packages online or write the expanded runtime to `/kaggle/working`.

```bash
kaggle-vllm bootstrap --profile kaggle-l4x4-cu129 --offline --runtime-source /kaggle/input/datasets/waqasm86 --native-wheel /exact/path/vllm-0.30.0-cp312-cp312-linux_x86_64.whl --strict
eval "$(kaggle-vllm env --profile kaggle-l4x4-cu129)"
kaggle-vllm doctor --profile kaggle-l4x4-cu129 --strict --json
```

Public Hugging Face delivery is implemented but deliberately refuses downloads until a published immutable commit is recorded in the L4 profile. No mutable `main` revision is trusted.

```python
from kaggle_vllm import KaggleLLM, KaggleSamplingParams, activate_runtime, discover_gemma4_qat

activate_runtime()
model = discover_gemma4_qat()
llm = KaggleLLM(model=model, profile="kaggle-l4x4-cu129", dtype="bfloat16", gpu_memory_utilization=0.86, max_model_len=2048)
outputs = llm.generate(
    ["Explain tensor parallelism in one sentence."],
    KaggleSamplingParams(temperature=0.0, max_tokens=64),
)
```

For Gemma's chat template, use `llm.chat([{"role": "user", "content": "Explain tensor parallelism in one sentence."}], sampling_params=KaggleSamplingParams(temperature=0.0, max_tokens=64))`. It tries `AutoProcessor.apply_chat_template` and then `AutoTokenizer`; neither path hard-codes model tokens.

For the explicit L4 profile, TP=4, `enforce_eager=True`, and `disable_custom_all_reduce=True` are P31 compatibility defaults, not performance maxima. Expert users can pass upstream options through `KaggleLLM(...)` and access `llm.upstream`. The wrapper does not force FlashAttention.

## Local verification

Use the user's Python 3.11 for SDK development. Do not import the cp312 native wheel locally.

```bash
PYTHONPATH=src python3.11 -m pytest -q
python3.11 -m build --no-isolation
```

See [compatibility](docs/compatibility.md), [runtime architecture](docs/architecture.md), [native evidence](docs/native-runtime.md), [doctor](docs/doctor.md), and [fresh acceptance](docs/next-kaggle-acceptance.md).

Runtime-v3 archive SHA256: `0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73` (6,633,730,070 bytes; 215 members).

---

## Preserved historical v0.2 documentation

The section below is retained verbatim from the v0.2 repository state for historical users. Commands and defaults above describe the v0.3 candidate.

# kaggle-vllm — Run vLLM on Kaggle platform with dual NVIDIA T4 GPUs

**kaggle-vllm** is a lightweight compatibility and runtime-delivery toolkit for running **upstream vLLM on Kaggle Notebooks with NVIDIA Tesla T4 GPUs**.

It is designed for users searching for a reliable way to **install vLLM on Kaggle**, run **LLM inference on Kaggle GPUs**, use **dual-T4 tensor parallelism**, validate **CUDA / PyTorch compatibility**, benchmark real multi-GPU behavior, persist vLLM sharded checkpoints, and launch an **OpenAI-compatible vLLM server** inside Kaggle.

> Current public SDK version: **0.2.0**  
> Validated profile: **Kaggle Notebook · 2 × NVIDIA Tesla T4 · SM75 / compute capability 7.5 · CUDA 12.8 · PyTorch 2.10 · CPython 3.12**  
> Upstream runtime source: **vLLM v0.18.1** at commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`

[Documentation](docs/README.md) · [Installation](docs/installation.md) · [Compatibility](docs/compatibility.md) · [Benchmarks](docs/benchmarking.md) · [Research](research/README.md) · [Troubleshooting](docs/troubleshooting.md)

---

## Research paper

**Characterizing Tensor-Parallel Crossover in vLLM Serving on Dual NVIDIA
Tesla T4 GPUs** — Mohammad Waqas (2026).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23119478.svg)](https://doi.org/10.5281/zenodo.23119478)
[PDF](research/paper/kaggle-vllm-research-paper.pdf) ·
[Zenodo](https://zenodo.org/records/23119478) ·
[engineering companion](https://waqasm86.github.io/posts/characterizing-tensor-parallel-crossover-vllm-dual-t4/) ·
[research methodology](research/README.md)

- TP=2 is not automatically faster than TP=1; crossover depends on model,
  exact token workload and offered concurrency.
- Prefill-heavy work reached favorable TP2 throughput earlier for the three
  fully measurable non-Qwen models; short work required much higher concurrency.
- Qwen had no sustained favorable short-workload point in the tested grid.
  Its prefill-heavy TP2 condition at concurrency 64 was resource-gated, not
  zero throughput.
- M3 measures PHB/NCCL communication independently; it does not establish
  NCCL alone as the cause of the serving crossover.

The paper characterizes the tested pinned runtime, workload, topology and
resource envelope; it does not claim a universal TP2 speedup.

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

**It depends on the model, exact token workload and offered concurrency.**
The published M4 study has five terminal logical repetitions per model and
workload on the pinned dual-T4 stack. Valid performance comparisons use five
matched repetitions. Its first sustained favorable TP2
**output-throughput** points are:

| Principal model | Balanced | Prefill-heavy | Short |
|---|---:|---:|---:|
| Qwen2.5-3B-Instruct | 64 | N/A: resource-gated at 64 | None in tested grid |
| Phi-4-mini-instruct | 32 | 4 | 64 |
| Ministral-3-3B-Instruct | 16 | 4 | 32 |
| Llama-3.2-3B-Instruct | 16 | 4 | 32 |

At Qwen prefill-heavy concurrency 64, TP2 exceeded the frozen resource ceiling;
there is no valid terminal throughput value, rather than zero throughput.
TP=2 may still help capacity when it does not improve performance.

The [M1 OPT-125M diagnostic](artifacts/kaggle-2026-09-01-milestone-1/)
measured lower TP2 output throughput at low load: 26.73% lower in graph mode
and 44.78% lower in eager mode. The
[M2 Qwen C=16 crossover](artifacts/kaggle-2026-09-02-milestone-2/) remains an
exploratory single-matrix observation. M4's repeated paired-repetition study
is the stronger final paper authority; the results describe different
evidence levels and workloads, without isolating a single causal mechanism.

Read the [paper PDF](research/paper/kaggle-vllm-research-paper.pdf),
[Zenodo DOI](https://doi.org/10.5281/zenodo.23119478),
[research methods](research/README.md),
[benchmark methods](docs/benchmarking.md) and
[concurrency methods](docs/concurrency-benchmarking.md).

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

Milestone 2 adds a separate exploratory online streaming concurrency benchmark
for one controlled Qwen TP1/TP2 matrix. On that pinned workload, TP2 crossed
TP1 in output throughput at concurrency 16; no
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
