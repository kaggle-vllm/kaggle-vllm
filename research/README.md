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
| M4 — multi-model crossover | IN_PROGRESS | Five-model gate closed: four passes and one canonical Gemma negative; four-model principal matrix ready for Kaggle |
| M5 — GuideLLM external validation | INCOMPLETE / POST-M4 | Prepared 30-shard Qwen balanced cross-check; Vidur source absent and no simulation fabricated |

The research questions and evidence vocabulary are defined in
[experiment_protocol.md](experiment_protocol.md). Run notebooks only in the
order in [KAGGLE_EXECUTION_CHECKLIST.md](KAGGLE_EXECUTION_CHECKLIST.md).

Machine-readable control files:

- `model_matrix.json`: pinned model/access/license/compatibility decisions.
- `m4_protocol.json`: M4 token workloads, matrix, repetitions and crossover rule.
- `M4_EXECUTION_PLAN.json`: exact compatibility, principal and GuideLLM shard order.
- `M4_SOURCE_FREEZE.json`: reviewed source/notebook/tool hashes; not GPU evidence.
- `M4_EVIDENCE_STATUS.json`: accepted compatibility ledger and next principal shard.
- `M4_GEMMA3_DIAGNOSTIC_REVIEW.json`: diagnostic history and final p13 canonical closure.
- `M4_PRINCIPAL_EXECUTION_QUEUE.json`: active principal queue plus auditable Gemma skips.
- `M4_PRINCIPAL_EXECUTION_GUIDE.md`: one-shard download and ingestion procedure.
- `M4_BATCH_PROTOCOL_AMENDMENT.md`: retained M4-BATCH-1 amendment.
- `M4_BATCH_EXECUTION_PLAN.json`: retained M4-BATCH-1 repetition batches.
- `M4_FILL_R00_ATTEMPT_1_REVIEW.json`: first batch-attempt forensic record.
- `M4_BATCH_PROTOCOL_AMENDMENT_V2.md`: M4-BATCH-2 continuation rules.
- `M4_BATCH_EXECUTION_PLAN_V2.json`: deterministic model-scoped continuations.
- `M4_BATCH_EXECUTION_GUIDE.md`: batch notebook, evidence, and ingestion procedure.
- `M4_BATCH_SOURCE_FREEZE.json`: retained M4-BATCH-1 source freeze.
- `M4_BATCH_SOURCE_FREEZE_V2.json`: M4-BATCH-2 notebook/runner/plan freeze.
- `HISTORICAL_QWEN_SHARDED_STATE.md`: scope of the external Qwen persistence evidence.
- `M5_DECISION.md`: post-M4 GuideLLM gate and claim boundary.
- `../scripts/generate_m4_queue.py`: deterministic active-queue generator.
- `python -m kaggle_vllm.research ingest-m4`: fail-closed download audit and local staging.
- `python -m kaggle_vllm.research ingest-m4-batch`: outer-bundle audit and independent inner-shard staging.
- `RESEARCH_MANIFEST.json`: research files, evidence sources and status gates.
- `PUBLICATION_READINESS.md`: explicit paper-readiness gates and blockers.
- `tooling_feasibility.md`: exact GuideLLM, Nsight and Vidur decisions.
- `claim_boundaries.json`: machine-readable allowed and unsupported paper claims.
- `model_artifact_workflow.md`: audited save/reload/API/upload/cleanup stages.

No generated number should enter a paper table manually. The scripts in
`analysis/` read committed machine-readable evidence and emit figures/tables.
The M4 runner and local assembler are prepared. Five models entered the frozen
compatibility population. Qwen, clean-rerun Phi, clean Llama 3.2, and text-only
Ministral passed. The final hash-verified p13 Gemma ZIP and source-equivalent
executed notebook establish the canonical, narrowly scoped
`UNSUPPORTED_DTYPE_INTERSECTION_ON_SM75_FROZEN_STACK`. Two standalone
Qwen-short repetitions and one batch-orchestrated Qwen-balanced shard are
preserved. The Qwen prefill-heavy r00 resource-gated attempt remains
review-required, and the remaining principal matrix and M5 independent
validation are not complete. The principal study population is
the four compatibility-passing models; Gemma remains a negative compatibility
result with N/A throughput, not throughput zero. The earlier Phi
notebook-drift candidate remains rejected
and is not combined with the accepted rerun. Llama's access-pending and token-
authorization failures remain access records, while its manually edited retry
remains excluded `DEBUG_COMPATIBILITY_PASS` evidence and is not combined with
the accepted clean run. Gemma 4 is an out-of-scope follow-up and was not
substituted post hoc.
