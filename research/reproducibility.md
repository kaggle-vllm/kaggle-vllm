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
`42bf096c032e2c6be1e2fa3d573c7c86ac589ba2`. The clean Qwen, clean-rerun Phi,
clean Llama 3.2, and clean Ministral compatibility shards are accepted under
`artifacts/kaggle-2026-09-08-milestone-4/compatibility/qwen25_3b/` and
`artifacts/kaggle-2026-09-08-milestone-4/compatibility/phi4_mini/`,
`artifacts/kaggle-2026-09-08-milestone-4/compatibility/llama32_3b/`, and
`artifacts/kaggle-2026-09-08-milestone-4/compatibility/ministral3_3b_bf16/`;
M4 remains in progress without an accepted principal matrix.
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

The accepted Ministral compatibility ZIP SHA256 is
`516e37c160ca98a49795870aeeeeb8a381cd8780924f4d9d8836305ad68c77cc`;
the executed notebook SHA256 is
`5f988e506a35a21f59c1502ae77eb5c6df9118549d4cd95ce7f0ea8a397c3a1d`.
The retained prompts are text-only. Although the multimodal-capable native
implementation performs encoder warmup and Transformers warns about the
legacy Mistral regex, an independent pinned-revision audit found identical
token sequences and exact counts for all 64 prompts with the regex correction
enabled.

The accepted Llama 3.2 compatibility ZIP SHA256 is
`37288c24065ddd2383c5e8dfebf08b61ca589bb2b8a37e9ae253ef9e6a1f9db4`;
the executed notebook SHA256 is
`cf12cd6c829896f50ceaa5dcd71ea8c0fb9465ceacaa701069df5167e4157de5`,
and the separately downloaded runtime JSON SHA256 is
`f6f15d4998e56c9acd7390ef222d8dc6bcd9bb8148149d1bf507f66c106a0577`.
The source-identical executed notebook used the ordinary evidence root and
clean implementation commit `42bf096c032e2c6be1e2fa3d573c7c86ac589ba2`.
Earlier `ACCESS_PENDING_AT_EXECUTION` and `TOKEN_AUTHORIZATION_BLOCKED`
attempts remain access records. The manually edited successful retry remains
excluded `DEBUG_COMPATIBILITY_PASS` evidence and is not promoted or averaged.

The Gemma diagnostic review is recorded in
`M4_GEMMA3_DIAGNOSTIC_REVIEW.json`. Three independently verified ZIPs have
SHA256 values
`f560479c6f5b119a17a7dbb5ce582e7fc72623bd586c2755f1e36e7e76f0bcdf`,
`9c3e77dba701d4f051d1edcab59431759630bab9a3494cf68e69c2e9c725d67a`,
and `0d975c5027df188bc3633cab50f98b3dfdcc43e923c97be0f97cb3a303b960f5`.
Each preserves the same TP1 and TP2 FP16 model-config rejection with no issued
requests. Available executed-notebook SHA256 values are
`bea77a4c2e633be3d6ab6c65829dca7f94da3beec5dee9037580ce160a97fff8`
and `54e53c09925ff45a2791edadef85d7e4ed6bda2509735f64d21b38fc06cfcac1`;
both contain replacement diagnostic execution code, while the middle ZIP has
no available notebook. These attempts are therefore diagnostic only. A single
fresh `compat-gemma3_4b` run from the unchanged frozen notebook is required to
canonicalize the expected negative result; it must not change dtype or attempt
to make Gemma pass.
