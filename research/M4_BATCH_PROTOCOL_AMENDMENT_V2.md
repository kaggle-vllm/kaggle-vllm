# M4 batch continuation amendment M4-BATCH-2

Date adopted: 2026-09-11 UTC

Status: adopted after the first real `fill-r00` attempt under `M4-BATCH-1`.

## Observed reason for this revision

The historical `M4-BATCH-1` plan remains unchanged. Its first `fill-r00`
allocation preserved `qwen25_3b-balanced-r00`, then stopped on the independent
14,848 MiB per-GPU guard in `qwen25_3b-prefill_heavy-r00`. Nine later logical
shards were not executed. The failed shard is retained as review-required
resource-boundary evidence and is not relabeled as a success.

`M4-BATCH-2` changes physical scheduling only. Future batches are scoped to one
model and one repetition. This limits one stopping shard to at most the rest of
that model/repetition group, while avoiding a return to 60 manual launches.
Groups are generated deterministically from the machine-readable principal
queue and the retained `M4-BATCH-1` ordering. They are not handwritten and do
not use throughput results to reorder scientific conditions.

## Continuation rules

- Every continuation has a new batch ID and a new physical Kaggle session ID.
- One continuation contains exactly one repetition and one model.
- Logical shard IDs, repetition indices, and within-shard behavior are unchanged.
- Already canonical logical shards are explicit skips.
- Failed review-required shards are explicit exclusions, not successes or skips.
- Not-executed shards remain queued until a later evidence-bearing allocation.
- Valid completed inner shards from a partial outer batch remain independently
  ingestible and preservable.
- Failed inner evidence remains review-only and cannot be promoted by batch
  ingestion.

## Unchanged scientific design

The principal design remains four models, three workloads, five repetitions,
TP1 and TP2, six concurrency points, 60 logical shards, and 720 fresh-server
cells. Request counts, exact token targets, model and tokenizer revisions,
dtype, decoding, resource ceilings, telemetry, metrics, and per-cell server
lifecycle are unchanged. The per-GPU ceiling remains 14,848 MiB; it is neither
summed across GPUs nor tuned after observation.

Logical repetition, physical Kaggle session, continuation batch, and
within-session workload order are separate provenance fields. Shards sharing a
continuation allocation are not described as globally independent environment
realizations.

## Historical boundary

`M4-BATCH-1`, its plan, notebook/source freeze, and `fill-r00` attempt remain
historical provenance. `M4-BATCH-2` does not rewrite them. The historical Qwen
TP2 sharded-state archive remains auxiliary platform-capability evidence only
and is not an M4 model source.
