# GitHub v0.3.2 release readiness

A local SDK wheel, sdist, SHA256SUMS and release-note draft exist. The native wheel and 6.4 GB dependency wheelhouse should remain on Hugging Face. [Draft PR #39](https://github.com/kaggle-vllm/kaggle-vllm/pull/39) is open for source and release-plan review. No tag or GitHub release has been published. After the Hub pin and a rebuilt SDK are verified, attach the exact SDK files and checksums to a draft GitHub release. Publish only after the user authorizes that visible action. Mark GPU testing pending until manual evidence arrives.

## PR #39 follow-up

Rechecked on 2026-10-16: draft PR #39 is OPEN, cleanly mergeable, head `8f49cf662c3cb9c5494d2090aed12f745992b7cd`, base `05f2b2983a4965ab0f450d4b9490bf9f3c041b55`. There are zero reviews, zero review threads, zero review comments and zero issue comments. Its three CPU/package CI jobs passed for the prior head. The local publisher-source changes are not covered by those prior CI results until pushed. GitHub v0.3.2 release is absent. Do not merge or publish while the native rights, immutable runtime pin and final distribution checks are unresolved.
