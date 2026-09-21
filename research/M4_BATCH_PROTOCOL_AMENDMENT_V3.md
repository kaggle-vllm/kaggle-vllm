# Prospective M4 terminal-resource continuation amendment M4-BATCH-3

Date proposed: 2026-09-14 UTC

Implementation commit: `264fbdbab1cb9e1930f454034154e74f1336904a`

Status: activated for future post-V8 sessions by
`M4_BATCH_SOURCE_FREEZE_V9.json`. The V8 `r02-llama` evidence was downloaded,
reviewed, promoted, and reconciled under M4-BATCH-2 before activation.

## Purpose and historical boundary

The Qwen2.5-3B prefill-heavy repetitions r00 and r01 independently crossed the
frozen 14,848 MiB per-GPU ceiling at TP2/concurrency 64. Those are measured
terminal scientific resource outcomes, not CUDA OOMs and not software failures.
Their historical `STOPPED_ON_FAILURE` outer manifests, inner evidence, source
freezes, reviews, and ingestion behavior remain unchanged.

M4-BATCH-2 requires the outer runner to stop after every nonzero shard return.
M4-BATCH-3 narrowly changes that outer orchestration rule for future,
not-yet-executed batches. It does not reinterpret any historical run.

## Continuable terminal-resource contract

Continuation is disabled unless the frozen plan explicitly selects
`M4-BATCH-3`, the versioned terminal-resource contract, and the approved reason
`VRAM_RESOURCE_GUARD`. Return code 3 alone is never sufficient.

Before a later planned shard may start, all of the following must pass:

- the shard, model, workload, repetition, source commit, runtime, and batch
  provenance match the frozen plan and source authority;
- the complete evidence directory and inner ZIP exist and their manifests and
  hashes verify;
- `terminal-resource-gate.json`, the raw result, execution summary, failed cell,
  and resource ledgers agree on `FAILED_RESOURCE_GATE` /
  `VRAM_RESOURCE_GUARD`;
- the frozen 12-cell grid is preserved, exactly one cell is resource-gated, the
  offending physical GPU indices and measured MiB are ledger-supported, and
  the 14,848 MiB per-GPU threshold is unchanged;
- the failed cell records missing throughput as null, not zero, and no row
  reports CUDA OOM;
- when the resource monitor terminates the cell process group, the failed cell
  may record the consequent `connection_error` plus `server_exit` observations
  and a graceful server return code of zero only when
  `monitor_action=TERMINATE_CELL_PROCESS_GROUP`, the request ledger contains
  the expected 192 failed requests, and every other terminal-resource check
  passes;
- no semantic, compatibility, model-load, NCCL, unknown server-lifecycle, or
  evidence-integrity failure is present;
- post-shard cleanup reports no new compute PID and GPU memory at or below the
  frozen idle baseline allowance;
- the disk and wall-clock guards pass again for the exact next planned shard.

If any condition is absent, ambiguous, or invalid, the batch fails closed. The
resource-gated shard is never rerun inside the batch. A reviewed terminal shard
also remains a no-rerun terminal queue outcome during later reconciliation.

## Batch outcome semantics

When every planned logical shard is executed and the only noncanonical outcomes
are verified terminal resource gates, the outer manifest status is
`COMPLETED_WITH_TERMINAL_OUTCOMES` and the runner returns zero. This means the
orchestration completed without an operational or integrity failure; it does
not promote the resource-gated shard to canonical success. Ingestion continues
to require review and preserves the failed evidence without inventing
throughput.

Unknown failures, arbitrary nonzero exits, CUDA OOM, stale GPU processes,
cleanup failure, archive/hash/provenance failure, source or runtime drift,
disk exhaustion, wall-clock exhaustion, NCCL failure, model-load failure,
server-lifecycle uncertainty, and unclassified exceptions remain fail-stop.

## Real r02-Qwen observation and prospective correction

The V10 `r02-qwen` attempt produced a third independent Qwen prefill-heavy
TP2/concurrency-64 resource boundary. The immutable outer attempt retained
`STOPPED_ON_FAILURE` because the original CPU fixture modeled only
`connection_error` with no server-exit return code. The real monitor-terminated
cell correctly contained `connection_error` plus `server_exit` and a graceful
return code of zero. The validator rejected that real shape before renewed
continuation guards could run, so short and balanced were not executed.

The historical V10 bundle, status, hashes, and measurements remain unchanged.
The correction accepts only the fully cross-checked monitor-caused exit shape
described above; arbitrary server exits, bare return code 3, CUDA OOM, NCCL,
model-load, cleanup, disk, wall-clock, source, runtime, and integrity failures
remain fail-stop. This defect affected evidence-collection continuation, not
the scientific workload or measured model performance. The 14,848 MiB per-GPU
ceiling and all benchmark parameters remain unchanged.

## Unchanged scientific design

The amendment changes evidence-collection reliability only. It does not change
the scientific matrix, model or tokenizer revisions, five-model compatibility
population, four-model principal scope, three workloads, five repetitions,
60 logical shards, 720 fresh-server cells, deterministic shard order, TP size,
six concurrency points, prompt or output lengths, dtype,
`gpu_memory_utilization`, decoding, telemetry, request counts, metrics, or the
14,848 MiB per-GPU ceiling. Qwen prefill-heavy r02, r03, and r04 remain required
and TP2/concurrency 64 remains in every repetition.

The completed `r02-llama` execution remains governed exclusively by
`M4_BATCH_SOURCE_FREEZE_V8.json` and M4-BATCH-2. This amendment was not used to
restart, replace, relabel, or ingest that historical execution. It applies
only to batches launched from the later M4-BATCH-3 source freeze.
