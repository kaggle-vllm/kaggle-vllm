# M4 final empirical phase

## Frozen design audit

The historical M4 plan contains five models, three workloads, and five
repetitions: 75 principal shard IDs. Each shard contains six request-concurrency
points and both TP1 and TP2, for 12 independently restarted serving cells.

The compatibility gate is closed. Four models are principal-eligible, yielding
60 executable shards and 720 serving cells. Gemma's 15 planned shard IDs remain
in the frozen plan as `SKIPPED_BY_COMPATIBILITY_GATE`; they are not deleted and
have no throughput measurement.

The first run is:

```text
M4_SHARD_ID=qwen25_3b-short-r00
```

The complete active order is generated in `M4_PRINCIPAL_EXECUTION_QUEUE.json`
and `M4_PRINCIPAL_EXECUTION_QUEUE.md`.

## Cost estimate and protocol decision

Compatibility runs took roughly 5.5–6.9 minutes for two concurrency-one cells
when serving succeeded. A principal shard starts 12 servers and adds larger
request sets at concurrency up to 64. Before principal observations exist, a
planning range of 0.5–1.5 wall-clock hours per shard is more defensible than a
point estimate. Across 60 dual-GPU sessions this is approximately 30–90 session
hours, or 60–180 GPU-hours. This is a planning estimate, not measured M4 data.

No protocol reduction is recommended. The frozen robustness criterion uses at
least five matched independent repetitions per TP/concurrency cell. A one-pass
localization phase followed by result-conditioned replication would leave many
model/workload cells without comparable uncertainty and could make follow-up
selection depend on observed effects. The 60-shard design should remain the
principal minimum; only the already frozen transition-refinement rule may add
predeclared repetitions near an observed sign change.

## Completion and analysis

After every download, run `python -m kaggle_vllm.research ingest-m4` with the
executed notebook, ZIP, and runtime JSON. The command fails closed on notebook,
archive, payload-hash, source, runtime, model, token, and cell-grid drift and
stages a uniquely addressed local candidate. Reviewed promotion into canonical
artifacts remains deliberate; the command does not silently edit the scientific
ledger.

Once all required principal shards are reviewed, use
`scripts/assemble_m4_evidence.py`. The analyzer treats the fresh server/session
repetition—not individual requests—as the inferential unit, preserves raw
samples, and applies the frozen paired-mean 95% confidence-interval rule.

## M5 boundary

M5 currently combines an independent GuideLLM cross-check and a proposed Vidur
simulator study. The frozen GuideLLM queue has 30 single-cell shards: Qwen
balanced workload, concurrency 1/16/64, TP1/TP2, five repetitions. It is
prepared but GPU-unexecuted and should run only after the corresponding M4 Qwen
cells are accepted. Vidur cannot proceed because the intended source is absent;
no simulator result is required for the central measured TP-crossover claim.
If the manuscript retains an independent-validation claim, the GuideLLM work is
required. Otherwise M5 must be explicitly framed as future validation, not
quietly marked complete.
