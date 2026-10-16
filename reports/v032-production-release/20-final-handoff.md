# 0.3.2 release follow-up handoff

Status: **PUBLICATION BLOCKED**. The 0.3.2 source candidate is in the existing release worktree and draft PR #39. The public SDK installation path is not yet usable because the versioned Hub metadata/native namespace, immutable SDK revision, PyPI 0.3.2 and GitHub v0.3.2 release are absent.

The revised delivery avoids mirroring 168 third-party dependency wheels. All 168 exact locked files were located at original publisher indexes and checked by advertised SHA256 or a direct byte download (TorchVision). The local SDK now has a checksum-pinned publisher source map and an allowlisted, resumable downloader. The proposed Hub mutation is only two copies of already hosted native wheels plus 14 small metadata/index additions. Native compiled-object redistribution clearance and the user's explicit authorization remain mandatory before that public change. The prior full-mirror plan is superseded.

PR #39 was rechecked as OPEN, DRAFT, cleanly mergeable, head `c1259865726a2ba7edee39e1b82b5677597380eb`, base `05f2b2983a4965ab0f450d4b9490bf9f3c041b55`, zero reviews/comments/threads; three current CI jobs passed. Local follow-up: 426 CPU tests passed, one skipped; the two primary notebooks passed static checks and were not executed. The changes were pushed to the existing PR and its three CI jobs passed.

Historic T4 0.2.0 wheel SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`. Preserved L4 native wheel SHA256: `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`. Current candidate SDK checksums are in `release/v032-SHA256SUMS.txt`; they are not final until an immutable Hub revision is embedded.

Remaining blockers: native-wheel legal/provenance sign-off; explicit Hub publication authorization; Hub publication and verification; immutable pin; final SDK rebuild; approved PR merge; authorized GitHub/PyPI publication and verification; user-run Kaggle four-L4 acceptance. Do not ask the user to run the 0.3.2 public notebook while any public release prerequisite remains. Exact notebook paths and status are in `21-manual-notebook-paths.md` and `.json`.

After pushing, the existing PR #39 head is `c1259865726a2ba7edee39e1b82b5677597380eb`. All three CI jobs passed for this commit; the PR remains draft with no review approval. Its body now reflects the publisher-source plan.

The read-only publisher availability audit passed 168/168 expected Content-Length checks. Exact SHA256 checks remain enforced during download. The native-wheel object inventory is in `native-compiled-object-inventory.json`; it is evidence for, not completion of, redistribution clearance.
