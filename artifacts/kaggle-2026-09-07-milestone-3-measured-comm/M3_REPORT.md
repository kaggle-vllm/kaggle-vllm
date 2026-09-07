# Milestone 3 — Measured NCCL/PHB Communication Characterization

Status: **MEASURED_ON_KAGGLE_PENDING_REVIEW**

This report is valid only when generated from the checksummed raw two-rank NCCL observations in the same evidence directory. It does not claim that PHB/NCCL is the sole cause of M1/M2 application behavior.

## Measured all-reduce summary

| Payload (bytes) | Samples | Repetitions | Mean (µs) | Median (µs) | p95 (µs) | p99 (µs) | Effective GB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 500 | 5 | 123.028 | 103.664 | 176.160 | 202.816 | 0.008 |
| 4096 | 500 | 5 | 122.298 | 95.520 | 204.704 | 303.104 | 0.033 |
| 16384 | 500 | 5 | 129.937 | 108.880 | 198.720 | 247.840 | 0.126 |
| 65536 | 500 | 5 | 138.832 | 100.384 | 174.400 | 281.056 | 0.472 |
| 262144 | 500 | 5 | 182.049 | 166.416 | 232.960 | 280.992 | 1.440 |
| 1048576 | 500 | 5 | 382.399 | 354.688 | 395.264 | 1896.032 | 2.742 |
| 4194304 | 500 | 5 | 1126.368 | 1120.736 | 1156.416 | 1407.712 | 3.724 |
| 16777216 | 500 | 5 | 4196.605 | 4208.816 | 4240.608 | 4254.112 | 3.998 |
| 33554432 | 500 | 5 | 8344.962 | 8361.632 | 8418.112 | 8450.432 | 4.021 |
| 67108864 | 500 | 5 | 16649.264 | 16666.768 | 16736.065 | 16772.768 | 4.031 |

## Measured communication fit

Equation: `T_us(S)=measured_allreduce_intercept_us + wire_bytes_per_rank / beta_effective`

Ring volume: `wire_bytes_per_rank = 2*(P-1)/P*S; for P=2, wire_bytes_per_rank=S`

- measured_allreduce_intercept_us: 112.135079
- beta_effective_gb_s: 4.063916
- R²: 0.999985

The intercept is a configuration-specific measured all-reduce proxy. It is not classical transport alpha or universal PCIe latency.

## M1/M2 consistency comparison

| Scenario | TP1 output tok/s | TP2 output tok/s | TP2/TP1 | Reciprocal delta (ms/output token) | Payload | Evidence |
|---|---:|---:|---:|---:|---:|---|
| m1-opt125m-graph | 1921.169 | 1407.602 | 0.733 | 0.190 | unknown | UNSUPPORTED |
| m1-opt125m-eager | 312.079 | 172.319 | 0.552 | 2.599 | unknown | UNSUPPORTED |
| m2-qwen-c01 | 26.295 | 14.950 | 0.569 | 28.858 | unknown | UNSUPPORTED |
| m2-qwen-c04 | 76.870 | 55.208 | 0.718 | 5.105 | unknown | UNSUPPORTED |
| m2-qwen-c08 | 108.527 | 99.802 | 0.920 | 0.806 | unknown | UNSUPPORTED |
| m2-qwen-c16 | 138.749 | 174.272 | 1.256 | -1.469 | unknown | UNSUPPORTED |
| m2-qwen-c32 | 158.655 | 267.793 | 1.688 | -2.569 | unknown | UNSUPPORTED |
| m2-qwen-c64 | 177.820 | 311.939 | 1.754 | -2.418 | unknown | UNSUPPORTED |

## Limitations

- The intercept is specific to this NCCL/runtime/process/topology/methodology configuration; it is not universal PCIe latency.
- A single linear model may not represent NCCL algorithm/protocol transitions across payload regimes.
- The fit uses payload-level means; uncertainty in raw collectives and independent repetitions is reported separately.
- The fit does not isolate PCIe, PHB, kernel launch, synchronization, or NCCL software contributions.
- Request concurrency is not instantaneous scheduler/decode batch size.
- Throughput-reciprocal deltas mix compute, scheduling, memory, communication, and serving effects.
- A fitted all-reduce value is consistency evidence, not proof that PHB/NCCL solely caused an application delta.
- Unexplained component is intentionally null because the application evidence does not isolate per-token service time.
