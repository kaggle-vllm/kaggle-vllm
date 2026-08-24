# Artifact policy

Large artifacts do not belong in Git. The validated wheel, Qwen archive,
safetensors, staged runtime, dependency overlay, caches, and extracted model are
excluded by `.gitignore`.

`kaggle-2026-08-23/` contains a small curated subset of the immutable local
Kaggle evidence. Binary distribution must use GitHub Releases, Hugging Face, or
another large-file artifact service and must retain checksums, compatibility
metadata, licensing, and upstream attribution.
