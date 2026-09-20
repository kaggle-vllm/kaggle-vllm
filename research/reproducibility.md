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
`artifacts/kaggle-2026-09-08-milestone-4/compatibility/ministral3_3b_bf16/`.
The canonical Gemma negative is under
`artifacts/kaggle-2026-09-08-milestone-4/compatibility/gemma3_4b/`;
M4 remains in progress without an accepted principal matrix. The original
one-shard/fresh-session path remains frozen. Dated amendment `M4-BATCH-1`
separately freezes a repetition-batched orchestration path and records the
shared Kaggle allocation as a blocking variable. Its first `fill-r00` attempt
is retained as a one-completed, one-resource-gated, nine-not-executed partial
batch. `M4-BATCH-2` keeps the scientific matrix fixed and narrows future
continuations to one model and one repetition per new physical allocation. Its
`fill-r00-phi` continuation preserves three independently canonical Phi r00
shards from one explicitly recorded physical session. The subsequent
`fill-r00-llama` continuation likewise preserves all three Llama r00 shards in
session `m4-fill-r00-llama-20260912T044444Z-dd577e28`; its outer ZIP SHA256 is
`9f27731cbaad5c43c52f56228534f8ef4f12d293681f10a8d2b67ee0ad7d4531`.
The `fill-r00-ministral` continuation preserves all three Ministral r00 shards
in session `m4-fill-r00-ministral-20260912T091607Z-54baa1e9`; its outer ZIP
SHA256 is
`5cc192952f00f11b37ff2b681e1c9eff24d7c285d85707dab989e3dd648798d2`.
The `fill-r01-phi` continuation preserves all three Phi r01 shards in session
`m4-fill-r01-phi-20260912T163631Z-eb65994e`; its outer ZIP SHA256 is
`5cf366ab2e284f0950e201578795ba9fba145b23343d77f9939711f52382e95f`.
The source-clean pre-audit executed notebook SHA256 is
`e65ca0bec247152d4fb64f9a41531e117d130800617cc296448469484dbbec36`.
The later notebook copy appends a read-only post-run audit and preserves the
same execution outputs, but is auxiliary because its saved benchmark-cell text
also contains a post-execution insertion. The independent runtime SHA256 is
`49a70b4ad76603eef7d86467f8f59aa3056824a4422139640b1a06bf90f48c7f`.
The `fill-r01-llama` continuation preserves all three Llama r01 shards in
session `m4-fill-r01-llama-20260913T141158Z-e929dba7`; its outer ZIP SHA256 is
`ced12dc3d498ecde6a894eed91caaa21585e14e350d519bea19e72016ae45e9e`.
Its executed notebook SHA256 is
`c69a5faafebd621560fb2bb67bc5b9ce13ae2c0cc1043c4f42b93afdb4461002`,
and the byte-identical separate/bundled runtime SHA256 is
`3c17bfda70bf97ad6d47a588f89157569e159d8a1b87ccb09a829f95258936b7`.
The `fill-r01-ministral` continuation preserves all three Ministral r01 shards
in session `m4-fill-r01-ministral-20260914T052525Z-750b6500`; its outer ZIP
SHA256 is
`d5d5a16ab89afa035a6ccf7ff8db80eeac9de06871dcbdbdac8bd5c3bae56947`.
Its executed notebook SHA256 is
`2f5c5b6fe91d3e01b08db3228a3fbd3ccc5ebaea0d245d339040a99478008d7a`,
and the byte-identical separate/bundled runtime SHA256 is
`7cddcfc1084a2b8627c900fed3c684305735db0b35a2f5e825b4c2e554fcaf53`.
The `fill-r01-qwen` continuation preserves balanced r01 and the terminal
prefill-heavy r01 resource boundary in session
`m4-fill-r01-qwen-20260914T083223Z-580bbec6`; its outer ZIP SHA256 is
`48e06604ef74680a04ff0be5eace719ff4667f635358b1fd16167542a50bdb3d`.
Its executed notebook SHA256 is
`1380f29e39f00e26e964ef288f852eba94f46d81838bedd0920af568862e1dcb`,
and the byte-identical separate/bundled runtime SHA256 is
`f0277b595ce4da1aa65b1a0b3c0f4a2f58246314f08eb4d1d529133bc8b05042`.
The notebook executable sources exactly match the V7 pin; the recorded stale
embedded digest is accepted only through that exact-source binding.
The V8 `r02-llama` continuation preserves all three Llama r02 shards in session
`m4-r02-llama-20260914T164332Z-3828ac3c`; its outer ZIP SHA256 is
`92e24500dc0d3b1003d485a1d64cf9431bb1e5e4727d35b62e7413a40e901a2b`.
Its executed notebook SHA256 is
`19fbc76ee01bcd301efb5f4a4a00cc422426b56e42491366820ea3f1f2dc9b2c`,
and the separate and bundled runtime files are byte-identical with SHA256
`b8fe083e9ae7536cf00214b225765482679f1efcd13765da57d326674f451f3b`.
The run is retained exclusively as V8/M4-BATCH-2 evidence.
Prospective amendment `M4-BATCH-3` (2026-09-14) distinguishes a verified
terminal scientific resource outcome from an outer operational failure. Its
implementation can continue to the exact next planned shard only when the
versioned gate record, source/runtime identity, inner hashes, failed-cell and
physical-GPU ledgers, null throughput semantics, cleanup, disk, and wall-clock
guards all pass. It does not change the matrix, workload, order, thresholds, or
historical r00/r01 artifacts. It was not retroactively applied to the V8
`r02-llama` execution. V9 activates it only for subsequent Kaggle sessions and
pins implementation commit `fc8f08ea76438159a7d3753066dc8c53146e5501`,
notebook pin `32e6a89d2f529d2087b82f05557fb371e1ce6d0e`, and the reconciled plan and queue.
The first real V9/M4-BATCH-3 execution, `r02-ministral`, preserves all three
Ministral r02 shards in session
`m4-r02-ministral-20260915T121550Z-233b4189`. Its outer ZIP SHA256 is
`f5ebd4db5d5ede4b30f04fd1ff07a99610d4c790f56ee0e2d38f5b5cb0af36d7`,
its executed notebook SHA256 is
`fd843f067a1fac2efb706253400ce7d7982f555ff0853e861c8ccfa753279230`,
and the separate and bundled runtime files are byte-identical with SHA256
`00eb16ef274e7ff57d8e466ed49792a686b441ccf732ac851445873ce364ea26`.
No terminal resource gate occurred, so the strict continuation-after-gate path
was not invoked; the outer order, cleanup, disk, and wall-clock guards passed.
V10 then pins the post-reconciliation clean notebook and unchanged M4-BATCH-3
contract for `r02-qwen`; V9 remains independently usable for the completed
Ministral evidence.
The V10 `r02-qwen` attempt preserved a third independent
`FAILED_RESOURCE_GATE / VRAM_RESOURCE_GUARD` result for prefill-heavy r02, with
11 valid cells and one TP2/concurrency-64 terminal row whose performance is
null, not zero. The immutable outer orchestration stopped because its CPU
fixture did not model the monitor-caused graceful `server_exit` record. The
fix applies only prospectively: V11 excludes the settled prefill-heavy shard
and schedules only short then balanced under the unchanged workload and
14,848 MiB per-GPU ceiling.
The V11 continuation completed in a second physical session,
`m4-r02-qwen-20260916T054736Z-95506297`, and executed only short then balanced.
Both are canonical. V10 and V11 remain distinct allocation records and combine
only through reviewed logical-shard reconciliation.
The V12 `r02-phi` continuation completed prefill-heavy, short, and balanced in
session `m4-r02-phi-20260916T141636Z-6f8069da`. All three 12-cell matrices are
canonical. Its outer ZIP SHA256 is
`3305ab862c47100e318d6b07b0245be2185a9a3def3d869f2ccf02e3c3fa3b2c`,
the executed notebook SHA256 is
`91f1626df2faa9122640373706927087115cc789c8ddc29495771a5185cff9bf`,
and the separate and bundled runtime files are byte-identical with SHA256
`badc544d5d074f7f4e626d62c446b4b2bb298855e01e6c53e666c248879c75e8`.
The notebook executable sources exactly match V12; cleanup returned both GPUs
to 0 MiB with no new compute PIDs and deleted only the exact Phi model cache.
All three Phi r02 shards are now in the reviewed no-rerun set.
V13 pins implementation commit `571bd07f478f820813ff6c9b19faf3470b138fb5`,
notebook pin `40c7c9da88bce712e3a87bcf9b5af52793cf6aa8`, and the post-Phi
queue/plan for `r03-ministral`; the freeze itself contains no GPU measurements.
The resulting session `m4-r03-ministral-20260919T094937Z-fee4d921` completed
all three 12-cell shards. Its outer ZIP SHA256 is
`723cf39bf30add663fb2492d6b2cdabb6beb103a95ae73f14322114055bedbc4`
and runtime SHA256 is
`117d1e58004975c2ff4972f0c7744f9afe081b9f12bc8589b1fa1a66ff2f15fc`.
The saved executed notebook SHA256
`064fc6977153956faba597274001f7027cc02f72677017eddc0819f4d32aeab3`
retains the original outputs and execution timestamps but has one runner-path
source edit made after execution. No clean pre-edit executed copy survives.
`M4_R03_MINISTRAL_NOTEBOOK_PROVENANCE_RECOVERY.json` binds that exact artifact
to immutable V13 and the final archive; ordinary source drift remains rejected.
V14 pins implementation commit `e290855de8aed29570cdd7952f9c2b37a2a19443`,
notebook pin `bd114f171f4ae9623b6e9538ed1789af90dce693`, and the
post-Ministral queue/plan for `r03-qwen`; it contains no new GPU measurements.
The resulting V14 session `m4-r03-qwen-20260919T152036Z-cd7c20ee`
preserved canonical short and balanced shards and a fourth independent
prefill-heavy terminal resource result. The outer ZIP SHA256 is
`27965628a214abad8d73b36f6cba2966a6cc41d97f020002e1be6e4591db8f60`,
the executed notebook SHA256 is
`8de8e45ad9bddbb770c824b0ab7dacfdba23b462d20c3790d7764da76156e1ef`,
and the byte-identical runtime SHA256 is
`935fba52f63fd653de8163ab777e5e9d68ab584841e5b8c32681439e96c03763`.
The prefill-heavy TP2/concurrency-64 cell reached 14,895 MiB independently on
both GPUs, did not report CUDA OOM, and retains null performance. A narrow
monitor-exit polling-race correction accepts its exact recorded exit shape
without weakening source, runtime, archive, or arbitrary-failure checks.
V15 pins implementation commit `2ea82d306177aaca70066542c4bd6195e205d67e`,
notebook pin `12e748a2c58db731b03dc28ba7477396cfe04f0f`, and the
post-Qwen queue/plan for `r03-phi`; it contains no new GPU measurements.
The resulting session `m4-r03-phi-20260920T045342Z-28f400e7` completed all
three 12-cell matrices. Its outer ZIP SHA256 is
`946aa7017302a9fe8893a5efc0a60e53ca452e6485ff2a9e254be416b5272d6e`,
executed-notebook SHA256 is
`8931ec2946ffdcb4cb1312cf0cadabe2bc9da9e274b3fbce633d7da85b234e58`,
and the separate and bundled runtime files are byte-identical with SHA256
`0971a403bb99d6750a5409a30059a51c5dca2672f051b1bb814a6c03bd1beadb`.
All three inner payload manifests, source/runtime/session bindings, request
ledgers, per-GPU resource guards, and cleanup checks pass. All three Phi r03
shards are now in the reviewed no-rerun set.
V16 pins implementation commit `45e1da603e4f27dca67168a9d4058c50a6c60556`,
notebook pin `da1395dd1c9e98968e3a29766069203be82aa4b6`, and the
post-Phi queue/plan for `r03-llama`; it contains no new GPU measurements.
The resulting session `m4-r03-llama-20260920T095447Z-7d3fc1d4` completed all
three 12-cell matrices. Its outer ZIP SHA256 is
`6707f48ca15e07430b022f49ba3531074e8031a9a89adbf68ae4eddb05721a54`,
executed-notebook SHA256 is
`ad852a7bf569f737f4a7d0f47a2e137cab410edc8fd3b3c5c0e0c6bd90c1b2a4`,
and the separate and bundled runtime files are byte-identical with SHA256
`945f095d723b4a602446f61a0604f8fae1ccd46715a017f4770dea0708446e6b`.
All three inner payload manifests, source/runtime/session bindings, request
ledgers, per-GPU resource guards, and cleanup checks pass. All three Llama r03
shards are now in the reviewed no-rerun set.
V17 pins implementation commit `2f12aeb4a1753edfbae427af9d147f07b58a2189`,
notebook pin `6a0637bc1dc3660ea9a3cb9918f951f5a444581f`, and the
post-Llama queue/plan for `r04-qwen`; it contains no new GPU measurements.
The resulting session `m4-r04-qwen-20260920T170853Z-0434d282` preserved
canonical balanced and short shards plus a fifth independent prefill-heavy
terminal resource result. Its outer ZIP SHA256 is
`7f690e99173420cc182573723a6c57daa62297c093b2b4c79f7b0f858d96cd4e`,
executed-notebook SHA256 is
`b5af736cba1fa6c812e0fbd319daad99e231472427ae9c5d360be4b691ad8793`,
and the separate and bundled runtime files are byte-identical with SHA256
`6d2b2cf63042391b112ecace5a1e9b83bad5441ed2094f71919eab0e9ec2fb17`.
The prefill-heavy TP2/concurrency-64 cell peaked at 14,895 MiB on each GPU,
crossed the frozen 14,848 MiB/GPU guard without CUDA OOM, and retains null
performance. Cleanup and renewed disk/wall-clock guards passed before short
completed, so all three r04 Qwen outcomes are in the no-rerun set.
V18 pins implementation commit `0ea6fed84f8679ee449f39312fc8f61dde75762e`,
notebook pin `9855590e758a7b70c9e4488f86af67b6fa3942eb`, and the
post-Qwen queue/plan for `r04-phi`; it contains no new GPU measurements.
The tracked reconciliation ledger excludes every canonical and reviewed
terminal shard from normal continuation. Batch ingestion permits
the runtime-to-later-shard interval only within the hash-frozen batch
wall-clock window; the standalone one-hour temporal rule remains unchanged.
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
whose checksum and semantic audits pass. The existing assembler accepts
multiple immutable execution-source freezes only when their frozen execution
plan, model matrix, and scientific protocol hashes agree. Verified terminal
VRAM rows remain resource-boundary observations with missing performance, never
zero throughput; the analyzer reports their repetition counts separately from
paired performance repetitions. `M4_EXECUTION_PLAN.json` fixes the principal
order; refinements are selected only after recording the observed transition
boundary. GuideLLM remains an independently versioned client at
commit `fc2dbe9edd4f7f1a4e9ccd752f6f43591adbcb73`.

Audit and stage each downloaded shard without editing the notebook:

```bash
PYTHONPATH=src /usr/local/bin/python3.11 -m kaggle_vllm.research ingest-m4 \
  --notebook /absolute/path/to/executed.ipynb \
  --evidence-zip /absolute/path/to/shard.zip \
  --runtime /absolute/path/to/runtime.json
```

The command refuses notebook-source drift, unsafe ZIP members, checksum drift,
source/runtime/model/token/grid mismatches, resource-limit violations, and
duplicate content. It stages a content-addressed candidate under excluded
`.local-evidence/`; canonical repository promotion remains a reviewed action.
The exact repetitive procedure and evidence-size policy are in
`M4_PRINCIPAL_EXECUTION_GUIDE.md`. Historical Qwen sharded-state capability
evidence is separately scoped in `HISTORICAL_QWEN_SHARDED_STATE.md` and is not
an M4 input.

Batch downloads use the same inner-shard semantic validator:

```bash
PYTHONPATH=src /usr/local/bin/python3.11 -m kaggle_vllm.research \
  ingest-m4-batch \
  --notebook /absolute/path/to/executed-batch.ipynb \
  --batch-zip /absolute/path/to/m4-batch-BATCH_ID.zip \
  --runtime /absolute/path/to/runtime.json
```

The outer archive and source freeze are checked first. Accepted inner shards
stage independently with execution mode, session ID, batch ID, repetition, and
within-session order. A later invalid shard does not erase earlier valid local
staging. Failed inner evidence remains review-only, not-executed rows remain
distinct from failures, and any partial outer batch returns review-required
status. A future M4-BATCH-3 outer bundle may report
`COMPLETED_WITH_TERMINAL_OUTCOMES` when every planned shard was attempted and
the only noncanonical outcomes are verified scientific resource gates. Such a
bundle still returns review-required at ingestion and never promotes missing
throughput to zero or a resource gate to canonical success.
Offline ingestion may independently verify a historical M4-BATCH-3 outer
`FAILED` row with return code 3 as a terminal resource candidate, but only by
passing the complete inner contract; it never rewrites the historical outer
manifest or treats a bare return code as sufficient.

The successful V11 outer ZIP SHA256 is
`6881890bd24fc9659d4e574ed122b2d77a5e377a4a2aa7f938d248753b7b8c13`;
the executed notebook SHA256 is
`dbbcf4cdd16cd886e4c21325d4532c999e68b7964d9d9a054344b7ed5f8044fe`;
the separate and bundled runtime are byte-identical with SHA256
`b41737b63f275be0412b16072a045b219a6b756957840f35a7348643fe5d5617`.
No Qwen r02 workload was duplicated and no new prefill-heavy measurement is in
the V11 bundle.

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

The Gemma diagnostic history and final closure are recorded in
`M4_GEMMA3_DIAGNOSTIC_REVIEW.json`. Three earlier independently verified ZIPs have
SHA256 values
`f560479c6f5b119a17a7dbb5ce582e7fc72623bd586c2755f1e36e7e76f0bcdf`,
`9c3e77dba701d4f051d1edcab59431759630bab9a3494cf68e69c2e9c725d67a`,
and `0d975c5027df188bc3633cab50f98b3dfdcc43e923c97be0f97cb3a303b960f5`.
Each preserves the same TP1 and TP2 FP16 model-config rejection with no issued
requests. Available executed-notebook SHA256 values are
`bea77a4c2e633be3d6ab6c65829dca7f94da3beec5dee9037580ce160a97fff8`
and `54e53c09925ff45a2791edadef85d7e4ed6bda2509735f64d21b38fc06cfcac1`;
both contain replacement diagnostic execution code, while the middle ZIP has
no available notebook. These attempts remain diagnostic only. Final p13 used a
source-equivalent frozen notebook; its ZIP SHA256 is
`6404b8abe22fc06211409288e59ae1d844354c1b12282a8c9140161c30347c12`, executed
notebook SHA256 is
`f1800e9be84fa297880960159da906612e1ef79d46b2c34344d27f0bbe2feee8`, and runtime
JSON SHA256 is
`8e747082b76d39852fa5838a3ce17f0ca8e392e7bac6ae4e0718700470e90ff5`.
Both TP modes reproduce the negative dtype boundary. No further Gemma run is
requested.
