# Communication Cost Modeling and Crossover Diagnostics

## Purpose

Milestone 3 turns existing M1/M2 measurements into a reusable diagnostics layer that:

1. Re-reads artifact JSON (no new GPU dependency).
2. Separates measured facts from derived proxies and optional hypothetical alpha-beta arithmetic.
3. Detects the measured TP throughput crossover on M2.
4. States explicitly when the analytical model cannot predict crossover from available evidence.

## Measured vs derived vs hypothetical

| Layer | Meaning | Example |
| --- | --- | --- |
| Measured | Directly taken from evidence JSON | TP1/TP2 tok/s, M2 matrix, summary crossover field |
| Derived proxy | Function of measured rates plus an assumed collective count | excess ms/tok divided by 2*layers |
| Hypothetical | Optional alpha-beta expression with labeled assumptions | N*alpha_assumed; optional S using concurrency as batch (weak) |

The derived proxy is not classical alpha. M1 already notes that offline TP deltas do not isolate collective time from compute, scheduling, graph/eager mode, or memory effects. Eager vs graph proxies on OPT-125M differ by a large factor; that instability is reported, not hidden.

## Architecture assumptions (collective count only)

Megatron-style tensor parallel is assumed to issue two AllReduce-like syncs per transformer layer (attention output projection and MLP down-projection):

- OPT-125M: 12 layers => 24 collectives/token (not "24 layers").
- Qwen2.5-3B: 36 layers => 72 collectives/token.

These counts are structural assumptions for normalizing a proxy. They are not proof of the vendor collective schedule inside vLLM for every kernel path.

## Payload S and concurrency

Online M2 concurrency is the offered load parameter for the benchmark harness. It is not established as the instantaneous decode batch size inside the vLLM scheduler. Therefore the default diagnostics path sets payload mode to unknown and does not compute beta*S from concurrency. A hypothetical mode may set payload_mode=hypothetical_concurrency_as_batch only with an explicit warning.

## Crossover: detection vs prediction

- Detection: first matrix concurrency where measured TP2 output tok/s exceeds TP1 (M2: c=16 under the pinned workload).
- Prediction: not supported with current evidence (no isolated collective timestamps; no trusted per-step batch size trace). The report records prediction_status=unsupported rather than implying the model predicted c=16.

## Topology language

topology.txt on the Kaggle dual-T4 host records PHB between GPUs. Diagnostics may mention that observation. They must not claim PHB/NCCL as the sole cause of TP behavior.

## CLI

Run:

PYTHONPATH=src python -m kaggle_vllm.diagnostics --m1-dir artifacts/kaggle-2026-09-01-milestone-1 --m2-dir artifacts/kaggle-2026-09-02-milestone-2 --output-dir artifacts/kaggle-2026-09-03-milestone-3 --format both

Optional flag --enable-hypothetical attaches labeled what-if numbers; it does not calibrate transport parameters from M1/M2.

## Non-goals

- No new vLLM/Torch/CUDA package dependency for import.
- No overwrite of historical M1/M2 artifacts.
- No silent equation of concurrency with decode batch.
