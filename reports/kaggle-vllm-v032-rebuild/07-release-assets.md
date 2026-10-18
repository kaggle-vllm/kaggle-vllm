# Release assets

Local candidate files: `dist/v032-rebuild/kaggle_vllm-0.3.2-py3-none-any.whl` and `dist/v032-rebuild/kaggle_vllm-0.3.2.tar.gz`. Their measured SHA256 values are in `RELEASE_MANIFEST.json` and `SHA256SUMS.txt`. The separate 293,855,397-byte native wheel retains SHA256 `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`; it was not modified or repackaged.

Planned draft release tag `v0.3.2`, title `kaggle-vllm v0.3.2 — NVIDIA L4 ×4 / Python 3.12 / vLLM 0.30.0`, status `PENDING_KAGGLE_ACCEPTANCE`. Requested assets: SDK wheel, sdist, native wheel, SHA256SUMS, RELEASE_MANIFEST, SOURCE_IDENTITY, COMPATIBILITY and installation README. No GitHub draft or Hub upload was created. Redistribution clearance is blocked, 168 dependency wheels are absent from the Hub, and no immutable v0.3.2 commit exists. Publishing a release description claiming HF-only install would be misleading.

Existing GitHub v0.3.2 tag/release was absent at audit time. Earlier releases and assets were not changed. The native wheel must be a release asset, never a Git commit.
