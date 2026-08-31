# kaggle-vllm

`kaggle-vllm` is a lightweight compatibility and runtime-delivery toolkit that
makes a validated upstream vLLM CUDA runtime usable on Kaggle's NVIDIA Tesla T4
environment without replacing Kaggle's preinstalled PyTorch/CUDA stack.

It validates the host and dependency baseline, resolves an immutable native
wheel, verifies its SHA256, stages it with `pip --target --no-deps`, creates a
separate dependency overlay and exposes small wrappers for inference,
persistent sharded state and upstream vLLM's OpenAI-compatible server.

It is **not** a vLLM fork, inference implementation, official Kaggle product or
official vLLM distribution. CUDA kernels, scheduling, tensor parallelism,
checkpoint persistence and serving are upstream vLLM capabilities.

> Current public PyPI SDK and repository version: **0.2.0**.
> Final published-package dual-T4 acceptance: **PASS on 2026-08-31**.
> Focused existing-Qwen TP=2 `sharded_state` regression: **PASS on
> 2026-08-31**. The earlier `0.2.0.dev0` acceptance and controlled benchmark
> remain separate 2026-08-30 evidence.

## Validated profile

| Dimension | Recorded value |
|---|---|
| Platform | Kaggle Notebook, Linux/glibc 2.35 |
| Python / ABI | CPython 3.12.13 / cp312 |
| PyTorch / CUDA ABI | 2.10.0+cu128 / 12.8 |
| CUDA toolkit | 12.8.93 |
| Driver | 580.159.04; driver-reported CUDA maximum 13.0 |
| GPUs | 2 × Tesla T4, 15,360 MiB, compute capability 7.5 / SM75 |
| NCCL / CMake / GCC | 2.27.5 / 3.31.10 / 11.4.0 |
| Upstream source | vLLM v0.18.1, `a26e8dc7ff2111a005144d775ecf9cebf56c45b2` |

The wheel distribution version is
`0.18.2.dev0+ga26e8dc7f.d20260822.cu128`; that generated version does not mean
the source was upstream v0.18.2. See [compatibility](docs/compatibility.md) and
[provenance](docs/provenance.md).

## Quick start

Install the released lightweight SDK on Kaggle:

```bash
python -m pip install "kaggle-vllm[hub]==0.2.0"
kaggle-vllm bootstrap --strict --dry-run
kaggle-vllm bootstrap --strict
eval "$(kaggle-vllm env)"
kaggle-vllm doctor --strict
```

Importing `kaggle_vllm` has no download, installation or activation side
effect. Bootstrap is explicit. It never installs vLLM, Torch or CUDA as normal
SDK dependencies.

Run inference through the thin wrapper:

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
    ["NCCL enables"], SamplingParams(temperature=0.0, max_tokens=32)
)
```

The defaults `dtype="float16"`, `enforce_eager=True` and
`disable_custom_all_reduce=True` are compatibility-first settings observed to
work on this profile, not universal performance recommendations.

Serve an upstream OpenAI-compatible endpoint on notebook-local loopback:

```bash
kaggle-vllm serve /kaggle/input/qwen2.5-3b-t4x2-sharded \
  --served-model-name qwen2.5-3b-kaggle-t4x2 \
  --load-format sharded_state \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70
```

## Large artifacts

Large files never belong in this Git history:

- native CUDA wheel: [Hugging Face binaries](https://huggingface.co/waqasm86/kaggle-vllm-binaries), immutable revision `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`, SHA256 `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`;
- Qwen TP=2 `sharded_state`: [Hugging Face model repository](https://huggingface.co/waqasm86/kaggle-vllm-models), archive SHA256 `12dcb264cb74e6fa2947b5f1fbebfa14562afa2292387f49e447b3290bc0b83b`.

The Qwen artifact is a vLLM-native, rank-specific TP=2 checkpoint—not a generic
Transformers checkpoint. It remains under the Qwen Research License, separate
from this SDK's Apache-2.0 license. TP=1, TP>2, topology-independent loading,
multi-node use and training are unvalidated or out of scope.

## What the evidence proves

Historical evidence proves native imports, raw two-rank NCCL, single-T4 and
dual-T4 OPT-125M inference, Qwen2.5-3B FP16 TP=2 save/reload and local OpenAI
models/completions/chat endpoints on the exact profile. On SM75, vLLM selected
`TRITON_ATTN`; FlashAttention 2 and SymmMem optimizations were unavailable, but
ordinary NCCL TP=2 still succeeded.

The controlled 2026-08-30 OPT-125M benchmark found TP=1 faster than TP=2 for
this tiny model. That result is consistent with communication/synchronization
overhead outweighing useful partitioned compute, but the historical benchmark
did not isolate one cause. TP=2 remains validated for capacity, NCCL execution
and topology compatibility, but it is not a universal performance improvement.
Other Kaggle GPU types, other Python ABIs for the native wheel, multi-node
operation, training and production readiness are not claimed.

The first post-0.2.0 engineering milestone adds CPU-testable, real-system TP
performance diagnostics. Preview a run without CUDA, vLLM, downloads or output
writes:

```bash
kaggle-vllm benchmark \
  --model facebook/opt-125m \
  --model-revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6 \
  --tensor-parallel-size 2 \
  --output /kaggle/working/tp2.json \
  --dry-run
```

Milestone 1 dual-T4 TP diagnostics were executed successfully on the validated
Kaggle T4 x2 profile. On the OPT-125M control, TP=2 reduced output throughput
by 26.73% in graph mode and 44.78% in eager mode relative to TP=1. Qwen2.5-3B
TP=2 with `max_num_batched_tokens=4096` did not demonstrate a meaningful
throughput improvement over baseline in the three-repeat workload. See the
[methodology](docs/benchmarking.md) and
[checksummed evidence](artifacts/kaggle-2026-09-01-milestone-1/).

## Documentation

Start with the [documentation index](docs/README.md):

- [installation and bootstrap](docs/installation.md)
- [architecture and design rationale](docs/architecture.md)
- [compatibility contract](docs/compatibility.md)
- [doctor and dependency baseline](docs/doctor.md)
- [native runtime and build provenance](docs/native-runtime.md)
- [multi-GPU/NCCL](docs/multi-gpu.md) and [persistent sharded state](docs/sharded-state.md)
- [OpenAI-compatible serving](docs/openai-serving.md)
- [benchmarking](docs/benchmarking.md) and [testing](docs/testing.md)
- [troubleshooting](docs/troubleshooting.md), [security](docs/security.md), and [release process](docs/release.md)

## Development

CPU development does not require vLLM or a GPU:

```bash
python -m pip install -e ".[test]"
pytest -q
ruff check src tests examples scripts
python -m build
python -m twine check dist/*
```

See [CONTRIBUTING.md](CONTRIBUTING.md). GPU acceptance is separate from CPU CI.
The final public-package and Qwen release evidence is summarized in the
[0.2.0 final acceptance report](docs/kaggle-v0.2.0-final-acceptance.md).

## License and attribution

The `kaggle-vllm` SDK is Apache-2.0. Upstream vLLM is an independent Apache-2.0
project. Qwen model materials are governed by their included Qwen Research
License. Project names and trademarks do not imply endorsement.
