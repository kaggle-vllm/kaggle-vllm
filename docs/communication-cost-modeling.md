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
| Hypothetical | Optional alpha-beta expression with labeled assumptions via Python API | N*alpha_assumed with explicit source_note |

The derived proxy is not classical alpha. Eager vs graph proxies on OPT-125M differ by a large factor; that instability is reported, not hidden.

## Architecture assumptions (collective count only)

- OPT-125M: 12 layers => 24 collectives/token (not "24 layers").
- Qwen2.5-3B: 36 layers => 72 collectives/token.

Unknown model labels/keys fail closed (no silent fallback to Qwen).

## Payload S and concurrency

Default diagnostics path does not treat request concurrency as instantaneous decode batch size.

## Crossover: detection vs prediction

- Detection: first matrix concurrency where measured TP2 tok/s exceeds TP1 (M2: c=16).
- Prediction: prediction_status=unsupported with current evidence.

## Topology language

topology.txt on the Kaggle dual-T4 host records PHB between GPUs. Diagnostics may mention that observation. They must not claim PHB/NCCL as the sole cause of TP behavior.

## CLI

```bash
PYTHONPATH=src python -m kaggle_vllm.diagnostics \
  --m1-dir artifacts/kaggle-2026-09-01-milestone-1 \
  --m2-dir artifacts/kaggle-2026-09-02-milestone-2 \
  --output-dir artifacts/kaggle-2026-09-03-milestone-3 \
  --format both
Supported flags:

--format md|json|both (controls which report files are written)
--no-strict-evidence (record structured errors instead of failing closed)
Hypothetical alpha/beta is available only via the Python API with explicit HypotheticalAlphaBetaAssumptions (no CLI flag; no baked-in 7.91 us / 7.8 GB/s defaults).

Non-goals
No new vLLM/Torch/CUDA package dependency for import.
No overwrite of historical M1/M2 artifacts.
No silent equation of concurrency with decode batch.