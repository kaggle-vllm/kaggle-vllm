# Artifact policy

Large artifacts do not belong in Git. The validated wheel, Qwen archive,
safetensors, staged runtime, dependency overlay, caches, and extracted model are
excluded by `.gitignore`.

`kaggle-2026-08-23/` contains a small curated subset of the immutable local
Kaggle evidence. Binary distribution must use GitHub Releases, Hugging Face, or
another large-file artifact service and must retain checksums, compatibility
metadata, licensing, and upstream attribution.

- `kaggle-2026-08-30-v0.1.2-benchmark/` — executed fresh-session
  `kaggle-vllm==0.1.2` TP=1/TP=2 OPT-125M benchmark evidence, including
  machine-readable results, logs, exact harness, runtime provenance and SHA256
  checksums.

- `kaggle-2026-08-30-v0.2.0-acceptance/` — executed fresh-session
  `0.2.0.dev0` Kaggle dual-T4 acceptance evidence, including the machine-readable
  acceptance record, OpenAI-compatible server log and executed-notebook checksum.

- `kaggle-2026-08-30-v0.2.0-benchmark/` — executed fresh-session
  `0.2.0.dev0` TP=1/TP=2 OPT-125M benchmark evidence, including machine-readable
  results, raw logs, exact benchmark harness, runtime manifest, provenance and
  SHA256 checksums.

Files under `artifacts/` are retained as immutable execution evidence. They are
excluded from source-formatting and linting tools so that post-capture tooling
does not modify their bytes or invalidate recorded checksums.
