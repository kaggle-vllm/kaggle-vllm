# kaggle-vllm 0.2.0 systems-research package

This directory turns the immutable kaggle-vllm 0.2.0 and upstream vLLM 0.18.1
runtime baseline into a reproducible research workflow. It does not define a
new package release.

## Milestone state

| Milestone | State | Evidence boundary |
|---|---|---|
| M1 — runtime/TP characterization | COMPLETE | Preserved Kaggle evidence, 2026-09-01 |
| M2 — Qwen serving crossover | COMPLETE | Preserved Kaggle evidence, 2026-09-02 |
| M3 — measured NCCL/PHB communication | COMPLETE | Canonical clean-source Kaggle evidence, 2026-09-07 |
| M4 — multi-model crossover | PREPARED_FOR_KAGGLE | Protocol, compatibility gates, and CPU analyzer; no M4 GPU results |
| M5 — simulator validation | OPTIONAL / SIMULATOR_COMPATIBILITY_LIMITATION | Required local Vidur source was not present |

The research questions and evidence vocabulary are defined in
[experiment_protocol.md](experiment_protocol.md). Run notebooks only in the
order in [KAGGLE_EXECUTION_CHECKLIST.md](KAGGLE_EXECUTION_CHECKLIST.md).

Machine-readable control files:

- `model_matrix.json`: pinned model/access/license/compatibility decisions.
- `m4_protocol.json`: M4 token workloads, matrix, repetitions and crossover rule.
- `M4_EXECUTION_PLAN.json`: exact compatibility, principal and GuideLLM shard order.
- `M4_SOURCE_FREEZE.json`: reviewed source/notebook/tool hashes; not GPU evidence.
- `RESEARCH_MANIFEST.json`: research files, evidence sources and status gates.
- `PUBLICATION_READINESS.md`: explicit paper-readiness gates and blockers.
- `tooling_feasibility.md`: exact GuideLLM, Nsight and Vidur decisions.
- `claim_boundaries.json`: machine-readable allowed and unsupported paper claims.
- `model_artifact_workflow.md`: audited save/reload/API/upload/cleanup stages.

No generated number should enter a paper table manually. The scripts in
`analysis/` read committed machine-readable evidence and emit figures/tables.
The M4 runner, local assembler, and GuideLLM cross-check are prepared, but no
M4/M5 GPU measurements exist yet.
