# M4 principal execution guide

Status: **IN_PROGRESS; 6 / 60 PRINCIPAL SHARDS PRESERVED**.

The canonical order is generated in `M4_PRINCIPAL_EXECUTION_QUEUE.json`. It
contains 60 active shard IDs and 15 non-executable Gemma historical rows. Use
one fresh Kaggle session per active row and change only the `M4_SHARD_ID`
secret. Never edit the notebook source.

## Repeat this procedure for one shard

1. Start a fresh Kaggle notebook session with **GPU T4 x2** and **Internet ON**.
2. Import the untouched `kaggle_vllm_m4_execute_shard.ipynb` from this
   repository. Its expected whole-file SHA256 is
   `885c5b6278614c66ac72f1e9ed8d05dbad84a19c8b99be129ebeb25e48a91dd9`.
3. Configure private Kaggle secrets `HF_TOKEN` and `M4_SHARD_ID`. The token is
   required by the frozen bootstrap even for an ungated model; never print it.
   Accept any gated model terms before its shard is scheduled.
4. Set `M4_SHARD_ID` to exactly one active queue value. Do not edit a cell to
   set the value.
5. Select **Run All**. Confirm the notebook prints source commit
   `42bf096c032e2c6be1e2fa3d573c7c86ac589ba2` and the intended shard.
6. Download the printed evidence ZIP. Its exact expected name is recorded as
   `expected_artifact` in the queue.
7. Download the executed notebook. This is the source-equivalence record, not
   a replacement for the frozen output-free notebook.
8. Separately download
   `/kaggle/working/kaggle-vllm-runtime/runtime.json`. This is the runtime-
   identity record. It is not inside the evidence ZIP.
9. Keep the original downloads in place. Compute local hashes, substituting
   browser-added filename suffixes when present:

   ```bash
   sha256sum <evidence-zip> <executed-notebook> <runtime-json>
   ```

10. From the repository root, audit and stage the three downloads without a
    network or GPU dependency:

    ```bash
    PYTHONPATH=src /usr/local/bin/python3.11 -m kaggle_vllm.research ingest-m4 \
      --notebook <executed-notebook> \
      --evidence-zip <evidence-zip> \
      --runtime <runtime-json>
    ```

11. Require `VERIFIED_CANONICAL_CANDIDATE` and, for an active shard,
    `PRINCIPAL_SHARD_PRESERVED`. The command stages a content-addressed local
    candidate under ignored `.local-evidence/m4-ingest/`; it does not promote
    a paper claim or modify canonical evidence.
12. Do not delete the Kaggle session until the local audit passes and the three
    original downloads are retained. Stop the session, then advance to the
    next active queue row.

The frozen hard VRAM ceiling is exactly **14.5 GiB = 14,848 MiB =
15,569,256,448 bytes per physical GPU**. The runner and local ingestion check
GPU0 and GPU1 independently; they neither use decimal 14.5 GB nor sum the two
devices against one ceiling.

## Current controlled shard

`qwen25_3b-short-r00`, `qwen25_3b-short-r01`, and
`qwen25_3b-balanced-r00` passed reviewed local preservation and ingestion.
`phi4_mini-short-r00`, `phi4_mini-balanced-r00`, and
`phi4_mini-prefill_heavy-r00` also passed reviewed batch ingestion.
`qwen25_3b-prefill_heavy-r00` remains failed/review-required after crossing the
per-GPU resource guard. For a standalone continuation, the selected queue row
is `M4_SHARD_ID=llama32_3b-short-r00` and its expected evidence ZIP is
`llama32_3b-short-r00-principal.zip`. Download the executed
`kaggle_vllm_m4_execute_shard.ipynb` and `runtime.json` alongside it. Validate
this shard end to end before advancing. The current preferred operation is the
model-scoped `M4_BATCH_ID=fill-r00-llama`; follow
`M4_BATCH_EXECUTION_GUIDE.md` instead.

## Completion gate

M4 is incomplete until all 60 active shards provide canonical, source-
equivalent, runtime-valid evidence. Every model/workload/repetition/TP/
concurrency cell must have the pinned model revision and prompt identity, the
expected request and token counts, zero unexpected request failure or OOM,
recorded resource telemetry, and a complete metric payload. An intentional
failure or skip must have an explicit machine-readable reason. Gemma's 15
historical principal rows remain `SKIPPED_BY_COMPATIBILITY_GATE` with
performance N/A.

## Evidence size and storage policy

The first principal ZIP measured 499,116 bytes and its 77 extracted members
measured 6,097,928 bytes. A simple 60-shard scale is about 30 MB compressed or
366 MB extracted before notebooks, runtime files, and audit records. Allow
**0.3--1.2 GiB** locally for model/workload variation and those additional
records, and continue replacing the planning range with observed totals.

This evidence is JSON, JSONL, logs, telemetry, manifests, checksums, and later
figures/tables. It is not a model checkpoint. Keep compact reviewed summaries
and justified evidence in Git only while repository size remains reviewable.
If the full canonical evidence would cause repository bloat, track immutable
SHA256-indexed summaries and retain the original evidence bundle in an
authorized external archive. No external upload is authorized by this guide,
and model weights, caches, wheels, and multi-GB checkpoint archives must not be
committed.
