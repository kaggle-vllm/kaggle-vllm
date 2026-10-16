# 0.3.1 to 0.3.2 migration

0.3.1 remains published unchanged. The 0.3.2 package changes its version, L4 staging root to `/tmp/kaggle-vllm/v0.3.2/l4x4`, and intended Hub namespace to `v0.3.2/l4x4`. The native vLLM 0.30.0 wheel and 168-wheel lock retain their bytes. The 195-wheel archival closure remains historical.

Public delivery now checks the entire immutable Hub tree against every pinned size and SHA256 before downloading; a resumable cache uses atomic activation, bounded retries and a lock. Runtime staging is serialized by a second lock. Strict CLI bootstrap checks host compatibility before large downloads and returns native diagnostics. `network-preflight` classifies common connectivity failures. The Hub revision is intentionally null until upload and verification.
