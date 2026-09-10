# M4 principal execution guide

Status: **READY_FOR_KAGGLE; NO PRINCIPAL RESULTS ACCEPTED**.

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

## First controlled shard

Run only `M4_SHARD_ID=qwen25_3b-short-r00` initially. The expected evidence ZIP
is `qwen25_3b-short-r00-principal.zip`. Download the executed
`kaggle_vllm_m4_execute_shard.ipynb` and `runtime.json` alongside it. Validate
this shard end to end before scaling the same procedure to the other 59.

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

The four successful two-cell compatibility directories occupy about 1.5--1.8
MB each. Scaling that observed compact evidence to 12 cells gives a planning
estimate near 10 MB per principal shard, or roughly 0.6 GB for 60 extracted
shards. Allow **0.3--1.2 GiB** locally for compressed ZIP variation, executed
notebooks, runtime files, audit records, and logs; replace this estimate with
the measured first-shard size before bulk execution.

This evidence is JSON, JSONL, logs, telemetry, manifests, checksums, and later
figures/tables. It is not a model checkpoint. Keep compact reviewed summaries
and justified evidence in Git only while repository size remains reviewable.
If the full canonical evidence would cause repository bloat, track immutable
SHA256-indexed summaries and retain the original evidence bundle in an
authorized external archive. No external upload is authorized by this guide,
and model weights, caches, wheels, and multi-GB checkpoint archives must not be
committed.
