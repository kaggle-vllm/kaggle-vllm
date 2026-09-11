# M4 repetition-batch execution guide

This guide applies amendment `M4-BATCH-1`. It reduces manual launches while
retaining all 60 logical shards and all 720 fresh-server serving cells. The
single-shard notebook remains valid for standalone execution.

## Frozen batches

Use one new Kaggle T4 x2 allocation for one batch ID. The five IDs are
`fill-r00`, `fill-r01`, `r02`, `r03`, and `r04`. They contain 11, 11, 12, 12,
and 12 remaining logical shards respectively. `qwen25_3b-short-r00` and
`qwen25_3b-short-r01` are recorded as already-canonical skips, not rerun.

Ordering is derived in `research/M4_BATCH_EXECUTION_PLAN.json` before bulk
collection. Model and workload orders are cyclic rotations of the actual active
lists keyed only by repetition. Shards remain grouped by model to reuse the
ordinary pinned checkpoint across that model's workloads. Ordering never uses
observed performance.

## Kaggle setup

1. Upload `kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb`.
2. Select two NVIDIA T4 GPUs, enable Internet, and use a new non-persistent
   allocation.
3. Configure private secrets `HF_TOKEN` and `M4_BATCH_ID` only.
4. Set `M4_BATCH_ID` to exactly one frozen batch ID and run all cells.
5. Do not edit the source cells, source commit, plan, order, revisions, runtime,
   memory threshold, or benchmark settings.

The notebook measures `/kaggle/working` with `shutil.disk_usage`, retains a
2 GiB safety reserve, and uses a 10.5-hour start cutoff with a 1.5-hour
per-next-shard margin. A guard stop finalizes valid completed evidence and marks
remaining shards `NOT_EXECUTED_IN_THIS_BATCH_ATTEMPT`; it is not a scientific
failure. A continuation requires a new physical session and a new session ID.

## Evidence and cleanup

Every logical shard keeps its own evidence directory, checksum manifest, and
ZIP. After each shard, hashes and the independent 14,848 MiB per-GPU guard are
checked. The runner waits for vLLM/Ray/NCCL child processes to terminate and
requires the GPUs to return to the recorded idle state before continuing.

The controlled model cache is `/kaggle/working/hf-cache`. Cleanup may delete
only the exact `models--OWNER--REPOSITORY` cache directory after every completed
workload for that model has hash-verified evidence and a ZIP. It never deletes
evidence, runtime files, source, the staged native runtime, the vLLM wheel, or a
broad cache root. If the exact path cannot be proven safe, cleanup does not run.

Download only:

- the printed `m4-batch-<batch-id>.zip`;
- the executed batch notebook;
- `/kaggle/working/kaggle-vllm-runtime/runtime.json`.

Do not download model weights, Hugging Face caches, the native wheel, or the
historical Qwen TP=2 sharded-state archive.

## Local ingestion

First preserve and hash the three browser downloads. Then run:

```bash
PYTHONPATH=src /usr/local/bin/python3.11 -m kaggle_vllm.research \
  ingest-m4-batch \
  --notebook <executed-batch-notebook.ipynb> \
  --batch-zip <m4-batch-BATCH_ID.zip> \
  --runtime <runtime.json>
```

The outer archive, source freeze, session identity, order, and every inner
shard are checked. Valid shards stage independently. Any invalid executed shard
makes the overall result `BATCH_REVIEW_REQUIRED` without deleting valid earlier
staging. No ingestion command promotes a paper claim automatically.
