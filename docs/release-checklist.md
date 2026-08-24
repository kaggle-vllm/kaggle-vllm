# v0.1 release checklist

## Source

- [x] Package version set to 0.1.0
- [x] Public wrapper, diagnostics, staging, sharding, and serving APIs documented
- [x] CPU tests and skip-marked GPU integration test added
- [x] Large binary/model/cache patterns ignored
- [x] Curated evidence provenance documented
- [ ] Final tests and Git size/secret review pass
- [ ] Feature branch pushed and PR opened for review

## GitHub prerelease

- [ ] Select unused RC tag (prefer `v0.1.0-rc1`)
- [ ] Create a prerelease, never overwrite an existing release
- [ ] Upload validated wheel
- [ ] Upload wheel SHA256 file
- [ ] Upload compatibility manifest
- [ ] Upload small final RC2 evidence archive if appropriate
- [ ] Verify release asset URLs and digest

## Hugging Face binaries

- [ ] Inspect/create `waqasm86/vllm-kaggle-binaries`
- [ ] Upload wheel using supported large-file transport
- [ ] Upload SHA256, compatibility manifest, and README

## Hugging Face Qwen sharded model

- [x] Archive SHA256 verified before extraction
- [x] Four expected TP rank/part files confirmed
- [x] Qwen Research License inspected
- [ ] Extract outside Git into a dedicated staging directory
- [ ] Retain Qwen LICENSE and add required NOTICE attribution
- [ ] Mark packaging/model-card modifications prominently
- [ ] State non-commercial use and TP=2 topology limitation
- [ ] Upload extracted usable directory, not only the tarball
- [ ] Verify files and repository visibility

Any authentication, checksum, licensing, or existing-release conflict blocks
only the affected publication step and must be reported exactly.
