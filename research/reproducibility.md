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
`42bf096c032e2c6be1e2fa3d573c7c86ac589ba2`. The corrected package remains
without accepted canonical M4 GPU evidence.
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
