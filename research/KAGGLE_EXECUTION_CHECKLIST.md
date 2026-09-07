# Exact Kaggle execution checklist

Use a new T4 x2 session for every numbered notebook. Internet must be enabled.
Never run model notebooks back-to-back in one session.

## 0. Publish and freeze the execution source

Push `research/m3-measured-comm-t4-phb-v020` without force and verify that the
remote branch resolves to the reviewed local commit. The prepared notebooks
clone that branch and record the exact resolved commit plus notebook hash in
their provenance. Do not start a Kaggle run when the remote ref differs from
the reviewed local commit.

## 1. M3 measured communication

Run `kaggle_vllm_milestone_3_measured_nccl_phb.ipynb` top to bottom. Download
the produced `kaggle-YYYY-MM-DD-milestone-3-measured-comm.zip`. It must contain:

- `M3_RAW_ALLREDUCE.csv`, `M3_RAW_ALLREDUCE.json`
- `M3_FIT.json`, `M3_FIT_RESIDUALS.csv`, `M3_FIT_RESIDUALS.svg`
- `M3_MODEL_VS_OBSERVED.csv`
- `M3_REPORT.md`, `M3_REPORT.json`
- `M2_SCHEDULER_SIGNALS.json`
- `M3_ENVIRONMENT.json`, `topology.txt`, `nccl-info.log`
- `gpu-telemetry.csv`, `M3_PROVENANCE.json`, `SHA256SUMS.txt`

Verify hashes locally, review measured-interval telemetry and NCCL identity, and
commit accepted evidence before creating the M4 branch.

## 2–5. One-model sharded-state sessions

Run in this order so lower-risk/public candidates provide early validation:

2. `kaggle_vllm_research_phi4_mini_t4x2_sharded.ipynb`
3. `kaggle_vllm_research_llama32_3b_t4x2_sharded.ipynb`
4. `kaggle_vllm_research_mistral_mid_t4x2_sharded.ipynb`
5. `kaggle_vllm_research_gemma3_4b_t4x2_sharded.ipynb`

Each evidence directory must contain `<slug>-t4x2-sharded.tar.gz.sha256`,
`<slug>-t4x2-sharded-manifest.json`, `<slug>-t4x2-validation.json`,
`<slug>-t4x2-provenance.json`, save/reload/API logs and JSON responses, and
`SHA256SUMS.txt`. For a permitted and verified upload, the large tarball and
local sharded/source state are removed only after remote object/size checks.
For Llama/Gemma, expect `UPLOAD_BLOCKED_LICENSE_REVIEW`; retain the local
artifact only as allowed by the applicable terms. An unsupported/OOM outcome
must contain the negative validation JSON and checksums.

## 6. M4 (only after M3 and artifact review)

Create `research/m4-multimodel-tp-crossover-v020` from the accepted M3 commit.
Include Qwen plus only candidates whose preceding validation passed. Execute the
matrix in `m4_protocol.json`, preserve every repetition and negative result, run
the standardized vLLM cross-check, then download raw JSON/JSONL, summaries,
server logs, telemetry, tokenizer prompt manifests, provenance and checksums.
The M4 execution notebook intentionally is not created on the M3 preparation
branch because real M3 evidence is a branch gate.

## 7. M5 (optional)

Do not execute until the exact intended Vidur source is supplied and inspected.
If mapping to vLLM 0.18.1 is invalid, publish
`SIMULATOR_COMPATIBILITY_LIMITATION` instead of a prediction.
