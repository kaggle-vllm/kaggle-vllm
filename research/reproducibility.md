# Reproducibility

The immutable native distribution is
`vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`
with SHA256
`5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`.
It derives from upstream vLLM v0.18.1 commit
`a26e8dc7ff2111a005144d775ecf9cebf56c45b2`. The strict 0.2.0 bootstrap stages
this wheel and a locked overlay without replacing Kaggle's Torch/CUDA stack.

Every evidence directory must carry provenance and `SHA256SUMS.txt`. Record the
repository commit/branch/dirty state, package and native identity, Python,
Torch, CUDA toolkit, driver, NCCL, GPU names/UUIDs, topology, model and tokenizer
revisions, exact command/config/seed/timestamps, notebook SHA, and output hashes.
Credentials and local usernames are excluded.

Reproduce in the order specified by `KAGGLE_EXECUTION_CHECKLIST.md`. M3 is
frozen under `artifacts/kaggle-2026-09-07-milestone-3-measured-comm/`; its
external ZIP and executed-notebook hashes are recorded in the directory README
and research manifest. M4 is based on post-M3 `main`. Its exact shard plan,
model/tokenizer revisions, primary runner, local checksum-validating assembler,
and isolated GuideLLM cross-check are frozen at source commit
`42bf096c032e2c6be1e2fa3d573c7c86ac589ba2`. The clean Qwen and clean-rerun
Phi compatibility shards are accepted under
`artifacts/kaggle-2026-09-08-milestone-4/compatibility/qwen25_3b/` and
`artifacts/kaggle-2026-09-08-milestone-4/compatibility/phi4_mini/`; M4 remains
in progress without an accepted principal matrix.
Generate the currently supported figures from the immutable evidence with:

```bash
/usr/local/bin/python3.11 research/analysis/generate_figures.py \
  --m1 artifacts/kaggle-2026-09-01-milestone-1 \
  --m2 artifacts/kaggle-2026-09-02-milestone-2 \
  --m3 artifacts/kaggle-2026-09-07-milestone-3-measured-comm \
  --figures research/figures \
  --tables research/tables
```

Pass reviewed M4/M5 evidence with the corresponding optional flags only after
it exists. Unavailable milestones are reported as unsupported rather than
synthesized.

For M4, retain the ZIP SHA, downloaded executed-notebook SHA, clean source
commit, prompt-manifest SHA, per-file checksums, server commands/logs, raw
requests, metrics, and resource samples for every shard. Assemble only shards
whose checksum and semantic audits pass. `M4_EXECUTION_PLAN.json` fixes the
principal order; refinements are selected only after recording the observed
transition boundary. GuideLLM remains an independently versioned client at
commit `fc2dbe9edd4f7f1a4e9ccd752f6f43591adbcb73`.

The accepted Qwen compatibility ZIP SHA256 is
`cd3c45dddf19830649b03a931cee0247d6f8b4ebbc953c433e3ade563b271ecb`;
the executed notebook SHA256 is
`d2106fcbd0df37cf34789499b90d71644c93b486d43265cc67c921b5ba91a94f`.
The earlier manually modified Qwen run remains excluded
`DEBUG_COMPATIBILITY_PASS` evidence and is never combined with canonical
measurements.

The accepted Phi compatibility ZIP SHA256 is
`7a20d7058cdcf7362704acbc5513db65eb865b864fd47b555f72465a32a0ebfe`;
the executed notebook SHA256 is
`878c9b980787acec04ede93071771465f9897acd412006afc60277d92893e3b9`.
The prior Phi candidate remains excluded `REJECTED_NOTEBOOK_DRIFT` evidence
because its executed notebook used `M4_SHARD_ID_2`. It is not promoted,
averaged, or copied into the canonical evidence directory.
