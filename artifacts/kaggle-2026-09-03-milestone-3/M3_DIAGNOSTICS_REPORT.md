# Milestone 3 - Communication-Cost Diagnostics Report

## Provenance

- Generated at (UTC): 2026-09-05T07:48:29.678264+00:00
- Generator git HEAD: `e4bddbc3ef8091c43791ad79e7618ee9bba1500e`
- Command: `python -m kaggle_vllm.diagnostics --format both`
- enable_hypothetical: False
- strict_evidence: True
- M1 dir: `artifacts/kaggle-2026-09-01-milestone-1`
- M2 dir: `artifacts/kaggle-2026-09-02-milestone-2`
- Output dir: `artifacts/kaggle-2026-09-03-milestone-3`
- Formats written: ['json', 'md']

## Scientific boundaries

1. Measured quantities come from M1/M2 JSON evidence.
2. Derived proxy = positive excess service time / assumed collectives; NOT isolated alpha.
3. Hypothetical alpha-beta requires explicit caller parameters (no baked-in 7.91/7.8).
4. PHB topology may be observed; it is not sole NCCL/PCIe causality.

## 1. Milestone 1

- M1 offline comparisons: treat graph vs eager proxy spread as a stability warning.
- M1 text already states communication was not causally isolated.
- OPT-125M excess-time proxy differs: graph~7.91 vs eager~108.29 us/collective (not stable transport alpha).

| Comparison | TP1 tok/s | TP2 tok/s | Delta % | Layers | Collectives/tok | Proxy us/coll | Signed dt ms/tok | Regime |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `comparison-opt125m-eager-tp1-vs-tp2` | 312.1 | 172.3 | -44.8 | 12 | 24 | 108.29 | 2.60 | TP2_SLOWER |
| `comparison-opt125m-graph-tp1-vs-tp2` | 1921.2 | 1407.6 | -26.7 | 12 | 24 | 7.91 | 0.19 | TP2_SLOWER |
| `comparison-qwen-tp2-batching` | 56.6 | 55.0 | -2.9 | 36 | 72 | n/a | n/a | INVALID |

- `comparison-opt125m-eager-tp1-vs-tp2`: MEASURED: TP1=312.0794 tok/s, TP2=172.3195 tok/s (-44.78%), regime=TP2_SLOWER. Assumed architecture: 12 layers, 24 collectives/token (2 per layer). DERIVED PROXY positive excess ~108.29 us/collective (not isolated alpha). PHB may be observed on host topology evidence; M1/M2 do not isolate NCCL/PCIe as sole cause of TP deltas.
- `comparison-opt125m-graph-tp1-vs-tp2`: MEASURED: TP1=1921.1690 tok/s, TP2=1407.6017 tok/s (-26.73%), regime=TP2_SLOWER. Assumed architecture: 12 layers, 24 collectives/token (2 per layer). DERIVED PROXY positive excess ~7.91 us/collective (not isolated alpha). PHB may be observed on host topology evidence; M1/M2 do not isolate NCCL/PCIe as sole cause of TP deltas.
- `comparison-qwen-tp2-batching`: MEASURED only for comparison-qwen-tp2-batching: not a controlled TP1/TP2 comparison. Do not derive collective excess-time proxy or PHB transport claims.

## 2. Milestone 2

- M2 online serving: concurrency is NOT proven equal to instantaneous decode batch.
- Crossover c=16 is MEASURED detection from the matrix, not an alpha-beta prediction.

| c | TP1 tok/s | TP2 tok/s | Delta % | Excess ms/tok | Signed dt ms/tok | Proxy us/coll | Regime |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 26.29 | 14.95 | -43.1 | 28.86 | 28.86 | 400.80 | TP2_SLOWER |
| 4 | 76.87 | 55.21 | -28.2 | 5.10 | 5.10 | 70.90 | TP2_SLOWER |
| 8 | 108.53 | 99.80 | -8.0 | 0.81 | 0.81 | 11.19 | TP2_SLOWER |
| 16 | 138.75 | 174.27 | +25.6 | n/a | -1.47 | n/a | TP2_FASTER |
| 32 | 158.65 | 267.79 | +68.8 | n/a | -2.57 | n/a | TP2_FASTER |
| 64 | 177.82 | 311.94 | +75.4 | n/a | -2.42 | n/a | TP2_FASTER |

## 3. Crossover: measured vs model prediction

- Measured throughput crossover concurrency: 16
- Measured rates: 138.75 -> 174.27 tok/s
- Model-predicted crossover concurrency: None
- Prediction status: `unsupported`
- Reason: M1/M2 lack isolated collective timestamps and trusted per-step scheduler batch sizes; alpha-beta cannot defensibly predict crossover.

## 4. Limitations

- Eager vs graph proxy divergence means proxy is not stable transport alpha.
- Serving concurrency is not silently treated as decode batch size.
- No sole NCCL/PCIe causality claim from M1/M2 alone.
