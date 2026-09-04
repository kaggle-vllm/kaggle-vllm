# Communication Cost Modeling & Crossover Diagnostics

## Overview
This document describes the analytical alpha-beta communication cost model implemented in `src/kaggle_vllm/diagnostics/`.

The model is designed to explain the low-concurrency Tensor Parallelism (TP2) penalty and the high-concurrency throughput crossover observed on Dual Tesla T4 GPUs connected via a PCIe Host Bridge (PHB).

## The Analytical Model (Alpha-Beta)

Total inter-GPU communication latency per generated token is modeled as:

$$T_{\text{comm}} = N_{\text{steps}} \times \left( \alpha + \frac{2(P - 1)}{P} \cdot \frac{S}{\beta} \right)$$

Where:
- $N_{\text{steps}}$: Number of AllReduce collective calls per token ($2 \times \text{layers}$ in Megatron TP).
- $\alpha$: Fixed per-collective synchronization latency across the CPU root-complex / PCIe bridge (inferred $\sim 7.91\,\mu\text{s}$ for SM75 PHB).
- $P$: Tensor Parallel degree ($P = 2$).
- $S$: Payload size in bytes ($S = \text{Batch\_Size} \times \text{Hidden\_Size} \times \text{Bytes\_Per\_Elem}$).
- $\beta$: Effective PCIe bus bandwidth ($\approx 7.8\,\text{GB/s}$ for PCIe Gen3 x8).

## Measured vs. Inferred Quantities

| Category | Variables | Source |
|---|---|---|
| **Measured** | Output tok/s, TTFT p95, TPOT p95, Peak Memory, Request Status | JSON Evidence Artifacts (`summary.json`) |
| **Inferred** | Step Alpha ($\alpha$), Comm Overhead ($T_{\text{comm}}$), Crossover Point | `AlphaBetaCommModel` Analytical Calculations |

## Limitations & Causal Boundaries

1. **Interacting Effects:** The model isolates collective communication latency, but measured throughput also includes vLLM scheduling overhead, CUDA Graph execution efficiency, and KV cache memory bus saturation.
2. **Topology Specificity:** The inferred step latency ($\sim 7.91\,\mu\text{s}$) applies strictly to PCIe Host Bridge (PHB) topologies without NVLink. On NVLink-enabled hardware, $\alpha$ drops significantly ($< 1.5\,\mu\text{s}$).
