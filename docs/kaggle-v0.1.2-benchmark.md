# kaggle-vllm v0.1.2 Kaggle dual-T4 benchmark

Date: 2026-08-30

Status: **EXECUTED**

This benchmark is post-release validation evidence for `kaggle-vllm==0.1.2`.
It does not modify the historical `v0.1.2` release or tag.

## Environment

- Kaggle Python: 3.12.13
- PyTorch: 2.10.0+cu128
- PyTorch CUDA: 12.8
- GPUs: 2 × NVIDIA Tesla T4
- Compute capability: 7.5 / SM75
- NCCL: 2.27.5
- Model: `facebook/opt-125m`

## Immutable native runtime

- Hugging Face repository: `waqasm86/kaggle-vllm-binaries`
- Revision: `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`
- Wheel: `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`
- SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

The Kaggle system Torch version, CUDA ABI and installation path were unchanged
before and after runtime activation.

## Controlled benchmark matrix

| TP | Eager | Custom all-reduce | Mean wall latency (s) | Mean output tok/s |
|---:|:---:|:---:|---:|---:|
| 1 | no | disabled | 0.248771 | 514.529814 |
| 1 | yes | disabled | 1.725596 | 74.190925 |
| 2 | no | disabled | 0.337742 | 379.268014 |
| 2 | yes | disabled | 3.087611 | 41.467032 |
| 2 | yes | enabled | 3.198598 | 40.027479 |

All five configurations completed with subprocess return code 0.

## Interpretation

For this small OPT-125M workload, TP=1 outperformed TP=2. This result should
not be generalized to larger models: the 125M model fits comfortably on one T4,
so tensor-parallel communication overhead can dominate useful computation.

Non-eager execution substantially outperformed eager execution in this
benchmark. Eager execution remains useful as a conservative functional
acceptance setting and should not be described as performance-optimal.

Enabling custom all-reduce did not improve the measured TP=2 eager workload on
this tested Kaggle dual-T4 topology. This is a workload-specific observation,
not a general claim about all T4 models or vLLM configurations.

## Evidence

Machine-readable JSON results, raw logs, the exact benchmark harness,
run metadata and checksums are retained under:

`artifacts/kaggle-2026-08-30-v0.1.2-benchmark/`

The executed notebook is:

`kaggle-notebooks/kaggle_vllm_0_1_2_benchmark.ipynb`

The earlier focused runtime-reset acceptance remains documented separately in:

`docs/kaggle-v0.1.2-reset-acceptance.md`
