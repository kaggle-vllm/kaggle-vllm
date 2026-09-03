# Kaggle Milestone 2 — Qwen TP1/TP2 concurrency crossover

## Execution identity

- Execution date (UTC): 2026-09-03
- Reviewed benchmark commit: `4f8dcc1c032d65d54b1cce3ca213535d68fd5099`
- Source-notebook orchestration fix: `9ef95c37d26ca04eb0156dc5b0e743e5885a8cd0`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Model revision: `aa8e72537993ba99e69dfaafa59ed015b17504d1`
- Native runtime: upstream vLLM `v0.18.1` at
  `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`, packaged as
  `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`
- Native wheel SHA256:
  `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`
- Original downloaded evidence ZIP SHA256:
  `e69c615d6b4d65d87c90448d37869c6dc47a84351c8823a698a46e45a158538f`

## Environment

The run used Python 3.12.13, PyTorch 2.10.0+cu128, CUDA toolkit/runtime 12.8,
NCCL 2.27.5, and two NVIDIA Tesla T4 GPUs with SM75 compute capability.
TP1 was restricted to physical GPU 0; TP2 used physical GPUs 0 and 1.

## Workload

The online OpenAI-compatible streaming matrix tested concurrency
`1, 4, 8, 16, 32, 64` with identical pinned model weights and request controls
for both TP degrees. Engine controls were `max_model_len=4096`,
`max_num_seqs=64`, and `gpu_memory_utilization=0.90`. Requests specified 512
output tokens and temperature 0. Prefix caching was disabled, and a fresh
server was launched for every TP/concurrency cell.

## Results

| Concurrency | TP1 output tok/s | TP2 output tok/s | TP2 delta | TP1 p95 TTFT (s) | TP2 p95 TTFT (s) | TP1 p95 TPOT (s) | TP2 p95 TPOT (s) | TP1 success/fail | TP2 success/fail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.29 | 14.95 | -43.1% | 1.606 | 0.950 | 0.0353 | 0.0663 | 20/0 | 20/0 |
| 4 | 76.87 | 55.21 | -28.2% | 5.862 | 3.223 | 0.0463 | 0.0697 | 20/0 | 20/0 |
| 8 | 108.53 | 99.80 | -8.0% | 11.696 | 6.293 | 0.0677 | 0.0792 | 24/0 | 24/0 |
| 16 | 138.75 | 174.27 | +25.6% | 21.906 | 11.876 | 0.1086 | 0.0887 | 48/0 | 48/0 |
| 32 | 158.65 | 267.79 | +68.8% | 43.826 | 23.503 | 0.1920 | 0.1138 | 96/0 | 96/0 |
| 64 | 177.82 | 311.94 | +75.4% | 84.778 | 46.288 | 0.3452 | 0.1988 | 192/0 | 192/0 |

## Findings

Under this pinned workload, TP1 delivered higher output throughput at
concurrency 1–8. TP2 first crossed TP1 at concurrency 16, then its measured
advantage grew at concurrency 32 and 64. All measured requests succeeded in
all 12 cells. Neither TP degree recorded a CUDA OOM, and no
TP1-fails/TP2-survives capacity crossover was observed through concurrency 64.

## Memory evidence

At concurrency 64, TP1 peaked at 14,255 MiB sampled on GPU 0. TP2 peaked at
14,701 MiB sampled on each T4, or 29,402 MiB (approximately 29.4 GiB)
aggregate sampled memory across the two devices. These are two separate T4
devices, not one 30 GB GPU. Sampled utilization can miss instantaneous peaks,
and high sampled memory is not an OOM; the request evidence and logs contain
no CUDA out-of-memory observation.

## Interpretation boundary

The measured crossover applies to this model revision, long-context prompt
profile, generation controls, vLLM runtime, and Kaggle T4×2 environment. It is
not a universal production threshold or proof that TP2 always wins at or above
concurrency 16. The PHB topology and sampled memory are observations, not
stand-alone causal explanations for the performance result.

## Reproducibility

[`summary.json`](summary.json) is the authoritative matrix and crossover
analysis. [`run-metadata.json`](run-metadata.json) records order, lifecycle,
model, runtime, and cleanup; [`environment.json`](environment.json) and
[`topology.txt`](topology.txt) preserve the platform observations.
[`SHA256SUMS.txt`](SHA256SUMS.txt) authenticates every generated evidence file.
Each cell also retains its result JSON, full request ledger, server log, raw
GPU telemetry, and before/after Prometheus snapshots.
