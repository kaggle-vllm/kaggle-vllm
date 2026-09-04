# Milestone 3 — Communication-Cost & Crossover Diagnostics Report

**Framework:** `kaggle-vllm` Analytical Alpha-Beta Diagnostics
**Scope:** Evaluation of PCIe Host Bridge (PHB) AllReduce penalty across M1 and M2 datasets

## 1. Milestone 1 — Low-Concurrency TP Penalty Analysis

| Comparison | TP1 (tok/s) | TP2 (tok/s) | Delta (%) | Inferred Step Alpha (µs) | Diagnostic Note |
|---|---|---|---|---|---|
| `comparison-opt125m-eager-tp1-vs-tp2` | 312.1 | 172.3 | -44.8% | 108.29 µs | TP2 is 44.8% slower due to PCIe PHB bus latency. Inferred AllReduce penalty is ~108.29 µs per step across 24 layers. |
| `comparison-opt125m-graph-tp1-vs-tp2` | 1921.2 | 1407.6 | -26.7% | 7.91 µs | TP2 is 26.7% slower due to PCIe PHB bus latency. Inferred AllReduce penalty is ~7.91 µs per step across 24 layers. |
| `comparison-qwen-tp2-batching` | 56.6 | 55.0 | -2.9% | N/A | MEASURED only: baseline vs max_num_batched_tokens variant (delta -2.9%). Not a controlled TP1/TP2 comparison — do not infer PHB AllReduce alpha. |

## 2. Milestone 2 — Concurrency Crossover Matrix

| Concurrency (c) | TP1 tok/s | TP2 tok/s | Delta (%) | TP1 TTFT p95 | TP2 TTFT p95 | System Regime |
|---|---|---|---|---|---|---|
| 1 | 26.29 | 14.95 | -43.1% | 1.61s | 0.95s | **COMMUNICATION-TAX** |
| 4 | 76.87 | 55.21 | -28.2% | 5.86s | 3.22s | **COMMUNICATION-TAX** |
| 8 | 108.53 | 99.80 | -8.0% | 11.70s | 6.29s | **COMMUNICATION-TAX** |
| 16 | 138.75 | 174.27 | +25.6% | 21.91s | 11.88s | **TP2-ADVANTAGE** |
| 32 | 158.65 | 267.79 | +68.8% | 43.83s | 23.50s | **TP2-ADVANTAGE** |
| 64 | 177.82 | 311.94 | +75.4% | 84.78s | 46.29s | **TP2-ADVANTAGE** |

> **Observed Throughput Crossover Point:** Concurrency **c = 16** (138.75 → 174.27 tok/s).

## 3. Causal Boundaries & Limitations

1. **Measured vs. Inferred:** Throughput and latencies are measured quantities from JSON evidence. Step alpha ($lpha$) and bus bandwidth ($eta$) are inferred analytical parameters.
2. **Non-Sole Causality:** While PCIe Host Bridge (PHB) latency explains the low-concurrency TP2 penalty, overall performance reflects interacting effects of scheduling, CUDA graph capture, memory bandwidth, and vLLM batching.
3. **Scope:** Results apply specifically to the pinned Kaggle Dual-T4 (SM75) environment and should not be extrapolated to NVLink or multi-node interconnects.
