# M4 batch protocol amendment M4-BATCH-1

Date adopted: 2026-09-11 UTC

Status: adopted after preservation of `qwen25_3b-short-r00` and
`qwen25_3b-short-r01`, and before bulk principal-matrix collection.

## Reason

The frozen M4 plan required one fresh Kaggle T4 x2 session for each logical
shard. Sixty manual notebook launches are operationally impractical and make
completion risk dominate the intended experiment. This amendment changes the
orchestration unit, not the scientific matrix.

## Amendment

Multiple logical shards may run sequentially in one Kaggle allocation through
the separately frozen batch notebook and runner. One batch contains only one
repetition index. No two repetitions of the same `(model, workload)` pair may
be intentionally executed in one allocation. The Kaggle allocation/session is
a blocking variable, and its identifier plus within-session order are required
provenance.

`qwen25_3b-short-r00` and `qwen25_3b-short-r01` remain standalone-session
observations. They are not retroactively described as batch executions. Their
allocation identifiers were not captured by the pre-amendment path and remain
explicitly unavailable rather than reconstructed.

## Unchanged scientific design

The following remain frozen:

- four principal models and their exact ordinary Hugging Face revisions;
- three workloads, five logical repetitions, TP 1 and TP 2, and concurrency
  1, 4, 8, 16, 32, and 64;
- request counts, exact-token prompt protocol, deterministic decoding, dtype,
  vLLM server configuration, metrics, and evidence fields;
- twelve serving cells per logical shard and a fresh vLLM server process for
  every serving cell;
- the 14,848 MiB hard ceiling applied independently to each physical GPU;
- 60 logical principal shards and 720 fresh-server serving cells.

Batch IDs are orchestration containers and never replace logical shard IDs.
Model load/cache time remains outside serving measurements, as before. No
measured throughput is changed or normalized by batch metadata.

## Added provenance and limitation

Every batch records a unique physical-session ID, batch ID, repetition, UTC
start/end, GPU names and UUIDs when available, runtime identity, ordered shard
IDs, actual execution order, per-shard UTC start/end, outcome, ZIP digest,
disk high-water mark, and batch wall clock. Each accepted logical shard records
`execution_mode=batch_orchestrated`, session ID, batch ID, repetition, and
within-session order.

Logical shards sharing a batch also share one physical Kaggle allocation and
are therefore not fully independent environment realizations. Methodology and
limitations must say this directly. The experimental unit for a condition's
confidence interval remains one logical repetition, never an individual
request. Session metadata is retained for blocking and sensitivity analysis;
it does not alter the frozen primary descriptive analysis.

## Historical Qwen sharded-state boundary

The approximately 4.77 GB historical Qwen TP=2 sharded-state archive remains
`AUXILIARY_PLATFORM_CAPABILITY_EVIDENCE` only. It is topology-aware, lacks
established TP=1 equivalence, and would confound TP with load format. Neither
the archive nor `waqasm86/kaggle-vllm-models` is an M4 model source. Both TP1
and TP2 continue to load the exact ordinary pinned Hugging Face checkpoint in
`research/model_matrix.json`.

## Relationship to the frozen single-shard path

The original `research/M4_EXECUTION_PLAN.json`, `research/m4_protocol.json`,
`research/M4_SOURCE_FREEZE.json`, single-shard notebook, and per-shard runner
remain historical frozen inputs. This amendment does not rewrite their
one-session-per-shard wording. It creates a second, independently frozen
`batch_orchestrated` provenance path while preserving `single_shard` ingestion.
