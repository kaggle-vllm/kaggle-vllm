# Historical v0.1 release checklist

## Source

- [x] Package version set to 0.1.2
- [x] Public wrapper, diagnostics, staging, sharding, and serving APIs documented
- [x] CPU tests and skip-marked GPU integration test added
- [x] Large binary/model/cache patterns ignored
- [x] Curated evidence provenance documented
- [x] Final tests and Git size/secret review pass
- [x] Feature branch pushed and PR opened for review

## GitHub prerelease

- [x] Select unused RC tag (prefer `v0.1.0-rc1`)
- [x] Create a prerelease, never overwrite an existing release
- [x] Upload validated wheel
- [x] Upload wheel SHA256 file
- [x] Upload compatibility manifest
- [x] Upload small final RC2 evidence archive if appropriate
- [x] Verify release asset URLs and digest

## Hugging Face binaries

- [x] Move/verify `waqasm86/kaggle-vllm-binaries`
- [x] Upload wheel using supported large-file transport
- [x] Upload SHA256, compatibility manifest, and README

## Hugging Face Qwen sharded model

- [x] Archive SHA256 verified before extraction
- [x] Four expected TP rank/part files confirmed
- [x] Qwen Research License inspected
- [x] Extract outside Git into a dedicated staging directory
- [x] Retain Qwen LICENSE and add required NOTICE attribution
- [x] Mark packaging/model-card modifications prominently
- [x] State non-commercial use and TP=2 topology limitation
- [x] Upload extracted usable directory, not only the tarball
- [x] Verify files and repository visibility

Any authentication, checksum, licensing, or existing-release conflict blocks
only the affected publication step and must be reported exactly.

## Historical 0.1.0 distribution

- [x] Merge the reviewed RC SDK PR into `main`
- [x] Add immutable native bootstrap and packaged overlay profile
- [x] Verify CPU-only bootstrap/download/activation tests on Python 3.11
- [x] Verify wheel/sdist metadata contains no vLLM, Torch, or CUDA dependency
- [x] Upload SDK wheel/sdist/checksums to Hugging Face and test immutable URL
- [x] Configure the PyPI pending Trusted Publisher
- [x] Publish and remotely verify `kaggle-vllm==0.1.0`
- [x] Verify `pip install kaggle_vllm` in a fresh environment
- [x] Merge the final bootstrap PR
- [x] Preserve `v0.1.0-rc1`; defer a final tag in favor of a version-aligned
  release after the 0.1.1 acceptance run

## 0.1.1 canonical-link maintenance

- [x] Move both Hugging Face repositories without re-uploading large artifacts
- [x] Preserve and verify the immutable native wheel revision and SHA256
- [x] Publish and remotely verify `kaggle-vllm==0.1.1`
- [x] Verify both normalized PyPI installation spellings in fresh environments
- [x] Publish the exact PyPI 0.1.1 SDK artifacts as an immutable HF fallback
- [x] Prepare an output-free Kaggle dual-T4 acceptance notebook
- [x] Execute and inspect the fresh Kaggle dual-T4 acceptance notebook
- [x] Approve final GitHub `v0.1.1` creation after acceptance documentation
  merges

## 0.1.2 safe runtime reset

- [x] Preserve normal non-empty destination refusal
- [x] Add manifest-aware reset planning and explicit `--yes` confirmation
- [x] Preserve cache and reject dangerous, overlapping, or symlinked paths
- [x] Add CPU-only reset safety tests
- [x] Publish and remotely verify `kaggle-vllm==0.1.2`
- [x] Verify both normalized PyPI installation spellings in fresh environments
- [x] Mirror the exact PyPI SDK artifacts to an immutable Hugging Face revision
- [x] Complete focused Kaggle reset/bootstrap/TP=2 acceptance
- [x] Create final GitHub `v0.1.2` from the accepted main commit

The active future release process is maintained in [release.md](release.md).

## 0.2.0 release preparation

- [x] Record and review 2026-08-30 development-candidate dual-T4 acceptance
- [x] Record and review the controlled OPT-125M TP=1/TP=2 benchmark
- [x] Correct benchmark versus acceptance source provenance in human docs
- [x] Complete final local test, package, clean-install and reproducibility checks
- [ ] Merge the reviewed 0.2.0 release-preparation PR
- [ ] Publish `kaggle-vllm==0.2.0` through PyPI Trusted Publishing
- [ ] Run fresh published-package Kaggle T4x2 acceptance
- [ ] Run the focused existing-Qwen TP=2 regression if required
- [ ] Merge final evidence, then create immutable `v0.2.0` tag and GitHub release
- [ ] Synchronize public Hugging Face cards with the published SDK artifacts
