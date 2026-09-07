# Exact Kaggle execution checklist

Use a new T4 x2 session for every numbered notebook. Internet must be enabled.
Never run model notebooks back-to-back in one session.

## Completed M3 source freeze and execution

M3 was executed from the reviewed clean source commit
`4df0dd183d78e48739f509637934adf33b98ce82`. The canonical ZIP SHA256 is
`b5aedfdda06522d483042e55a4f9637dd582e72242936ee130cddeeae0ae5769`; the
executed notebook SHA256 is
`a2bf24a37a63d12318c5188829310f95d8f3106733be5d301b5570bd63e18383`.
The accepted evidence is under
`artifacts/kaggle-2026-09-07-milestone-3-measured-comm/`. Do not rerun or
replace it when executing M4.

## M3 measured communication (COMPLETE)

Start one brand-new Kaggle session with Internet enabled and the T4 x2
accelerator. Import the repository
`kaggle_vllm_milestone_3_measured_nccl_phb.ipynb` unchanged and run it top to
bottom. Do not add or run any live source-patch cell. Confirm that the bootstrap
prints the reviewed PR #25 head SHA before continuing.

The run must validate CPython 3.12, two Tesla T4 devices, compute capability
7.5, PHB topology, PyTorch/CUDA/NCCL identity, kaggle-vllm 0.2.0, upstream vLLM
source identity, and native wheel SHA256
`5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`.
It must use 20 warmups, 100 timed collectives per payload/repetition, and five
fresh-process repetitions while retaining NCCL INFO logs and measured-interval
telemetry. The primary run must not force `NCCL_ALGO` or `NCCL_PROTO`.

The output directory and ZIP must be named
`kaggle-YYYY-MM-DD-milestone-3-measured-comm` with no `retry` suffix. Treat the
result as **CANONICAL_CANDIDATE** until its hashes and scientific content pass
review. Download both the ZIP and the executed notebook. The ZIP must contain:

- `M3_RAW_ALLREDUCE.csv`, `M3_RAW_ALLREDUCE.json`
- `M3_FIT.json`, `M3_FIT_RESIDUALS.csv`, `M3_FIT_RESIDUALS.svg`
- `M3_MODEL_VS_OBSERVED.csv`
- `M3_REPORT.md`, `M3_REPORT.json`
- `M2_SCHEDULER_SIGNALS.json`
- `M3_ENVIRONMENT.json`, `topology.txt`, `nccl-info.log`
- `gpu-telemetry.csv`, `M3_PROVENANCE.json`, `SHA256SUMS.txt`

Verify `SHA256SUMS.txt` before interpretation. Require provenance to record the
reviewed commit, `dirty: false`, source-notebook hash, full runner command,
runtime identity, timestamps, and artifact hashes. Review the complete grid,
fit, residuals, measured-interval telemetry, NCCL identity, resource ceilings,
and claim classifications. Compare the result with the preserved retry-1 debug
run only for plausibility; agreement does not make the debug run canonical.
The canonical review passed every listed gate. The retry1 package remains a
non-canonical `DEBUG_VALIDATION_RUN` comparison only.

## M4 source and session controls

Use the output-free
`kaggle_vllm_m4_execute_shard.ipynb`. Its pinned source commit must equal the
reviewed M4 source identity
`5176f1a133da0d8457937993be07f5a2a2b22a5f` printed by the notebook. For every shard:

1. start a brand-new Kaggle session;
2. select **GPU T4 x2** and enable Internet;
3. configure `HF_TOKEN` without printing it; accept a gated model's terms first;
4. set `M4_SHARD_ID` to exactly one ID from `M4_EXECUTION_PLAN.json`;
5. run the unmodified notebook top to bottom;
6. download the printed ZIP and the executed notebook;
7. stop the session before starting another model/shard.

The runner starts a fresh vLLM server for every TP/concurrency cell, disables
prefix caching, uses tokenizer-controlled `/v1/completions` prompts, requires
server-reported 128/512/2048 input tokens and 64/256/128 output tokens, and
preserves failures. It actively terminates a server on a 14.5 GiB/GPU or 28
GiB system-RAM guard breach and refuses projected `/kaggle/working` use above
20 decimal GB. A zero return code means `CANONICAL_CANDIDATE`, not accepted.

## M4 compatibility order

Run and audit these five two-cell shards in this exact order:

1. `compat-qwen25_3b`
2. `compat-phi4_mini`
3. `compat-llama32_3b`
4. `compat-ministral3_3b_bf16`
5. `compat-gemma3_4b`

For Llama and Gemma, lack of accepted gated access is an access result, not an
architecture incompatibility. For any model, a preserved unsupported/OOM
outcome is valid negative evidence. Do not run its principal shards unless the
compatibility gate passes.

## M4 principal order

For each compatibility-passing model in the same model order above, run:

1. workload order: `short`, `balanced`, `prefill_heavy`;
2. within each workload: repetitions `r00`, `r01`, `r02`, `r03`, `r04`;
3. within every shard the notebook fixes concurrency order to 1, 4, 8, 16, 32,
   64 and interleaves TP1 then TP2.

The 75 fully enumerated possible shard IDs and prerequisites are in
`M4_EXECUTION_PLAN.json`; models that fail compatibility reduce the executed
set. Audit every ZIP hash, source identity, prompt-manifest hash, exact token
counts, resource telemetry, failures, and `SHA256SUMS.txt`. Assemble only
accepted principal shards with:

```bash
PYTHONPATH=src /usr/local/bin/python3.11 scripts/assemble_m4_evidence.py \
  --output-dir /absolute/non-git/path/m4-assembled \
  /absolute/path/to/each/accepted/shard ...
```

If the signed TP2-minus-TP1 effect changes between adjacent principal points,
run only justified refinement points (12, 20, 24, or 48 as appropriate) with
repetitions 5–9 so those cells total ten independent repetitions. The runner's
exact form is `--mode refinement --concurrency C --repetition R`; record the
data-derived selection before execution.

## Optional sharded-state infrastructure

The four `kaggle_vllm_research_*_t4x2_sharded.ipynb` notebooks are optional
repeatability infrastructure, not a prerequisite for the Transformers-based M4
comparison. Run one only after its model compatibility and license review pass.
Llama/Gemma uploads remain `UPLOAD_BLOCKED_LICENSE_REVIEW`. Never commit model
archives to GitHub.

## GuideLLM independent cross-check

After accepting the matching Qwen M4 cells, run
`kaggle_vllm_m5_guidellm_crosscheck.ipynb` in new sessions. Set
`GUIDELLM_SHARD_ID` according to `M4_EXECUTION_PLAN.json`. The exact initial
cross-check covers balanced Qwen at concurrency 1, 16, and 64, TP1/TP2, five
repetitions each. The notebook pins GuideLLM commit
`fc2dbe9edd4f7f1a4e9ccd752f6f43591adbcb73` in a separate client virtual
environment. Download every ZIP and executed notebook. Do not equate GuideLLM
metrics with primary-client metrics until definitions and raw records reconcile.

## Vidur and optional profiling

No Vidur source exists locally. Do not produce simulator predictions until the
exact intended source is supplied and inspected. Current status is
`SIMULATOR_COMPATIBILITY_LIMITATION`. Nsight Python is auxiliary-only and must
not be installed in the canonical Kaggle runtime; use the already prepared CUDA
events, NCCL INFO, Prometheus metrics, and nvidia-smi telemetry instead.
