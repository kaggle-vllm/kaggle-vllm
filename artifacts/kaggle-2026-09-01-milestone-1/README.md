# Kaggle Milestone 1 dual-T4 TP diagnostics

Executed on 2026-08-31 UTC / 2026-09-01 local.

## Source identity

kaggle-vllm:
7e7355d64266e864a8113c30d52c612d98100350

Native vLLM source:
v0.18.1
a26e8dc7ff2111a005144d775ecf9cebf56c45b2

Native wheel SHA256:
5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c

Qwen model revision:
08bb62d0b68d20062e9009a9769c0df53d3dae21

Downloaded evidence ZIP SHA256:
6ed9a2697b36699840d42585db4f8af55ee3db32687121b0c07d27bdd02db1d9

## Environment

- Kaggle Notebook
- Python 3.12.13
- PyTorch 2.10.0+cu128
- CUDA toolkit 12.8.93
- NCCL 2.27.5
- Driver 580.159.04
- 2 × Tesla T4 / SM75
- GPU-to-GPU topology: PHB
- No NVLink token observed
- NVIDIA P2P read/write query: OK

## Results

| Configuration | Output tok/s |
| --- | ---: |
| OPT-125M TP1 graph/non-eager | 1921.17 |
| OPT-125M TP2 graph/non-eager | 1407.60 |
| OPT-125M TP1 eager | 312.08 |
| OPT-125M TP2 eager | 172.32 |
| Qwen2.5-3B TP2 baseline | 56.60 |
| Qwen2.5-3B TP2 `max_num_batched_tokens=4096` | 54.95 |

Observed deltas:

- OPT graph TP2 versus TP1: -26.73%
- OPT eager TP2 versus TP1: -44.78%
- Qwen 4096 versus baseline: -2.91%

The Qwen batching delta is small relative to the observed
three-repeat variability and is not treated as a robust regression.
`max_num_batched_tokens=4096` did not demonstrate a meaningful throughput
improvement in this workload.

## Interpretation

TP=2 is operational and uses both T4 GPUs. On the small OPT-125M control
workload TP=2 was materially slower than TP=1, so TP=2 is not a universal
throughput optimization.

The topology observation is consistent with communication and
synchronization overhead being relevant, but these measurements do not
isolate PCIe/NCCL as the sole cause.

Graph/non-eager execution was substantially faster than eager execution
for this workload.

## Limitations

These are offline `vllm.LLM.generate` measurements, not HTTP serving
benchmarks. Results apply only to the captured runtime and workloads. They do
not support arbitrary-GPU or multi-node extrapolation, training claims, or
proof of PCIe-only causality. Three Qwen repeats are insufficient for strong
conclusions about a small delta. Model-load timing is also confounded by
execution order and cache warming, so it is not attributed to the batching
setting.
