# Run first: kaggle-vllm 0.3.0 corrected split-v2 acceptance

Run this **only** in a brand-new Kaggle CPython 3.12/Linux x86_64/four-NVIDIA-L4 competition notebook with Internet disabled. This package has not passed its own fresh-session acceptance yet.

## Attach existing inputs

1. This small, private `kaggle-vllm-v030-production-acceptance` Dataset under `/kaggle/input/datasets/waqasm86/`. It carries the SDK wheel, runner/notebook, evidence schema, and no runtime dependency wheels. It contains no CUDA, Torch, native vLLM, model, or runtime-v3 payload. Public redistribution of the runtime wheels still requires manual license review.
2. Existing corrected `kaggle-vllm-v030-py312-runtime-core-v2`, `kaggle-vllm-v030-py312-runtime-large-v2`, `kaggle-vllm-v030-py312-torch-cu129-v2`, and `kaggle-vllm-v030-py312-wheelhouse-audit-v2` Datasets. Extracted directories and mounted ZIP files are supported. The P33 lean lock draws its 168 dependency wheels from these Inputs; the `build-deps-v2` Dataset is not required.
3. Preferred native source: save/version the successful P31 final notebook **with outputs**, then attach that Notebook Output. The runner discovers its exact vLLM wheel or `p31-kaggle-vllm-v030-l4x4-final-v2.tar.gz` under `/kaggle/input/notebooks/`. If its mount differs, pass `--notebook-root`, `--native-wheel`, or `--p31-archive` with an exact mounted path. A dedicated wheel-only Dataset is the fallback described in [P31 output reuse](../docs/p31-notebook-output-reuse.md).
4. Kaggle-provided Gemma Input Model `gemma-4-31b-it-qat-w4a16-ct`. The runner requires a single valid version under `/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/`, or an explicit `--gemma-model` path.

**The 6,633,730,070-byte runtime-v3 tar is not a prerequisite.** It remains an optional future-public-release source. Older non-v2 Datasets are ignored; duplicate corrected inputs fail closed. Every selected wheel is checked against the pinned runtime manifest and corrected wheelhouse-audit metadata before bootstrap. The native SHA256 must be `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`.

## Notebook or shell

Open `kaggle_vllm_0_3_0_l4x4_final_sdk_acceptance.ipynb` from this Dataset and run its single code cell. It defaults to `SPLIT_V2_KAGGLE_INPUTS` and runs Gemma only. Alternatively, in a Kaggle Bash cell:

```bash
set -euo pipefail
INPUT=/kaggle/input/datasets/waqasm86
mapfile -t RUNNERS < <(find "$INPUT" -type f -name 'production_acceptance_v030.py' | sort)
test "${#RUNNERS[@]}" -eq 1
ACCEPTANCE_DIR=$(dirname "${RUNNERS[0]}")
(cd "$ACCEPTANCE_DIR" && sha256sum -c SHA256SUMS.txt)
python3.12 -u "${RUNNERS[0]}"
```

For a mounted P31 tar outside the default Notebook Output root, append `--p31-archive '/kaggle/input/<exact-path>/p31-kaggle-vllm-v030-l4x4-final-v2.tar.gz'`. For a mounted direct wheel, append `--native-wheel '<exact-path>/vllm-0.30.0-cp312-cp312-linux_x86_64.whl'`. Component overrides and the optional runtime-v3 mode are in [fresh acceptance](../docs/fresh-kaggle-v030-acceptance.md).

The runner installs only the small SDK with `pip --no-index --no-deps`, stages the 168 SHA-verified dependency wheels and P31 native wheel to `/tmp/kaggle-vllm/v0.3.0/l4x4/`, and runs doctor, core/FA2 native imports, the libcuda link probe, NCCL world_size=4, and `KaggleLLM.chat()` semantic Gemma generation. Child gates stream saved logs, have timeouts, and terminate their process group on failure. It does not use the Kaggle API, download artifacts, compile CUDA, or rebuild vLLM.

## Download evidence

Download `/kaggle/working/kaggle-vllm-v030-production-acceptance/20_evidence_bundle.zip` and retain the executed notebook output. Review `evidence/input_inventory.json`, `evidence/05_split_source_integrity.json`, `logs/07_bootstrap.log`, `evidence/10_doctor.json`, `logs/15_nccl_tp4.log`, `evidence/17_gemma.json`, `evidence/19_final_matrix.json`, and `evidence/20_evidence_sha256.json`.

`FRESH_KAGGLE_PASS` needs all mandatory gates. `FA2_MODEL_RUNTIME_PASS` stays false: the competition run verifies built-in FA2 binary/import only. P31 Gemma selected `TRITON_ATTN`; unrelated-model FA2 testing belongs in a later permitted L4 environment. Public runtime redistribution remains blocked pending manual third-party license review.
