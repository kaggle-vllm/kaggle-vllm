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

Reproduce in the order specified by `KAGGLE_EXECUTION_CHECKLIST.md`. Do not move
to an M4 branch until M3 evidence has been downloaded, hash-verified and reviewed.
Generate the currently supported figures from the immutable evidence with:

```bash
/usr/local/bin/python3.11 research/analysis/generate_figures.py \
  --m1 artifacts/kaggle-2026-09-01-milestone-1 \
  --m2 artifacts/kaggle-2026-09-02-milestone-2 \
  --figures research/figures \
  --tables research/tables
```

Pass reviewed M3/M4/M5 evidence with the corresponding optional flags only
after it exists. Unavailable milestones are reported as unsupported rather than
synthesized.
