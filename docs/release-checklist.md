# v0.1 release checklist

## Source

- [x] Package version set to 0.1.0
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

- [x] Inspect/create `waqasm86/vllm-kaggle-binaries`
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

## Final 0.1.0 distribution

- [x] Merge the reviewed RC SDK PR into `main`
- [x] Add immutable native bootstrap and packaged overlay profile
- [x] Verify CPU-only bootstrap/download/activation tests on Python 3.11
- [x] Verify wheel/sdist metadata contains no vLLM, Torch, or CUDA dependency
- [x] Upload SDK wheel/sdist/checksums to Hugging Face and test immutable URL
- [ ] Configure the PyPI pending Trusted Publisher
- [ ] Publish and remotely verify `kaggle-vllm==0.1.0`
- [ ] Verify `pip install kaggle_vllm` in a fresh environment
- [ ] Merge the final bootstrap PR
- [ ] Tag the exact final main commit and publish GitHub `v0.1.0`
