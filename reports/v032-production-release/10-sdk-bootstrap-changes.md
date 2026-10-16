# SDK bootstrap changes

The CLI accepts `kaggle-vllm bootstrap --profile kaggle-l4x4-cu129 --strict --json`. It checks CPython cp312, Linux x86_64, Kaggle markers and exactly four NVIDIA L4/SM89 GPUs before fetching public data. A local-only 3.11 run can test parsing and integrity but cannot stage cp312 native files.

Public fetching is locked to one validated Hub revision and the packaged 168-wheel allowlist. It validates remote path, size and LFS digest, checks free disk, resumes a partial download with HTTP Range, verifies SHA256, and atomically renames completed files. It serializes cache access and SDK staging. Strict mode runs the native doctor and returns a JSON installation report. Public fetch remains fail closed while the immutable revision is null.

Known limit: Hugging Face delivery still needs a completed public upload and real Kaggle disk/network acceptance. The runtime overlay occupies substantial scratch space; cleanup uses the SDK-owned reset command after reviewing its manifest.
