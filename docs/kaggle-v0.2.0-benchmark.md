# kaggle-vllm 0.2.0.dev0 Kaggle dual-T4 benchmark

Date: 2026-08-30

Status: **EXECUTED**

This report records the fresh-session controlled benchmark of the reviewed
`kaggle-vllm==0.2.0.dev0` development candidate on Kaggle dual Tesla T4 GPUs.

The benchmark source state was:

- Git repository: `kaggle-vllm/kaggle-vllm`
- Source commit: `6d10912ad73e81f5a62fcec299c87ed5b2631b4f`
- SDK version: `0.2.0.dev0`

This is development-candidate evidence and does not claim that a final 0.2.0
release already exists.

## Environment

- Kaggle Python: 3.12.13
- PyTorch: 2.10.0+cu128
- PyTorch CUDA: 12.8
- GPUs: 2 × NVIDIA Tesla T4
- Compute capability: 7.5 / SM75
- NCCL: 2.27.5

The Kaggle system PyTorch installation was unchanged before and after runtime
activation.

## Native runtime

- Hugging Face repository: `waqasm86/kaggle-vllm-binaries`
- Immutable revision: `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`
- Native wheel:
  `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`
- SHA256:
  `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

The native runtime is the same artifact used by the previously validated
0.1.x line.

## Model identity

The benchmark used the original Hugging Face model:

- Repository: `facebook/opt-125m`
- Immutable revision:
  `27dcfa74d334bc871f3234de431e71c6eeba5dd6`

The model was downloaded once from Hugging Face and all five benchmark
configurations used the same local immutable snapshot.

## Controlled benchmark matrix

| TP | Eager | Custom all-reduce | Mean wall latency (s) | Mean output tok/s |
|---:|:---:|:---:|---:|---:|
| 1 | no | disabled | 0.2455 | 521.94 |
| 1 | yes | disabled | 1.7854 | 71.74 |
| 2 | no | disabled | 0.3538 | 363.26 |
| 2 | yes | disabled | 3.4277 | 37.34 |
| 2 | yes | enabled | 2.8500 | 44.91 |

All five configurations completed with subprocess return code 0.

Each configuration ran in a separate Python process so vLLM workers and GPU
allocations were released before the next configuration.

## Interpretation

For this small OPT-125M workload, TP=1 outperformed TP=2. This should not be
generalized to larger models. OPT-125M easily fits on one Tesla T4, so
tensor-parallel communication overhead can dominate useful computation.

Non-eager execution substantially outperformed eager execution for this tested
workload. The eager configuration remains useful as a conservative functional
acceptance setting and should not be described as performance-optimal.

The TP=2 eager custom-all-reduce-enabled result was faster than the
corresponding custom-all-reduce-disabled result in this run. The earlier
v0.1.2 benchmark showed a different small difference. Because each
configuration contains only three measured repeats, these small cross-run
differences should not be treated as statistically robust performance
improvements or regressions.

The strongest project-level conclusion is therefore that the 0.2 SDK and
runtime-delivery hardening preserved functional operation and did not introduce
an obvious catastrophic performance regression relative to the validated
0.1.2 baseline.

## Evidence

Machine-readable JSON results, raw logs, the exact benchmark harness, runtime
manifest, run metadata, and SHA256 checksums are retained under:

`artifacts/kaggle-2026-08-30-v0.2.0-benchmark/`

The executed benchmark notebook is:

`kaggle-notebooks/kaggle_vllm_0_2_0_benchmark.ipynb`

The benchmark evidence archive downloaded from Kaggle had SHA256:

`543c5601ceb7162ca9f9225b041a3a07f7b672bde625950b77834a7fcdf721c1`
