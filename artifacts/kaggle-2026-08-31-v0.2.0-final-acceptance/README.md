# kaggle-vllm 0.2.0 final Kaggle acceptance evidence

This directory contains the small machine-readable and log evidence reviewed
for the 2026-08-31 final release gates. The executed notebooks are retained at
their canonical paths under `kaggle-notebooks/` and are covered by
`SHA256SUMS.txt`.

The three Qwen files are byte-identical copies of the files downloaded from
the successful Kaggle run:

- `qwen-regression-evidence.json`
- `qwen-regression-started.json`
- `qwen-tp2-generation.log`

The original post-publication acceptance JSON, OpenAI server log, and runtime
manifest were not downloaded separately. The complete JSON object printed by
the executed acceptance notebook was recovered field-for-field into
`kaggle-vllm-020-published-acceptance-evidence.recovered.json`. It is recovered
notebook-output evidence, not an untouched raw Kaggle sidecar. Exact recovery
provenance is recorded in `evidence-provenance.json`.

No model weights, native wheel, CUDA library, Torch package, cache, or other
large runtime file is included.
