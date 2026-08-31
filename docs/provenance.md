# Source and artifact provenance

The machine-readable consolidated record is
[`artifacts/BUILD-PROVENANCE.json`](../artifacts/BUILD-PROVENANCE.json). Static
CI cross-checks it against the packaged profile, compatibility JSON and
`artifacts/SHA256SUMS.txt`.

## Native identity

- upstream repository: `https://github.com/vllm-project/vllm`
- source tag/commit: `v0.18.1` /
  `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`
- wheel distribution version: `0.18.2.dev0+ga26e8dc7f.d20260822.cu128`
- wheel SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`
- binary HF revision: `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`

The local upstream working tree used during the 0.2.0 audit had advanced to
commit `6a9c69fa851389dcf1ee5d3a2363e27af665d26d`; comparisons were made against
the immutable v0.18.1 tag object above. The upstream tree was not modified.

## Qwen identity and license boundary

The historical archive SHA256 is
`12dcb264cb74e6fa2947b5f1fbebfa14562afa2292387f49e447b3290bc0b83b`.
The public model repository was observed at revision
`08bb62d0b68d20062e9009a9769c0df53d3dae21` on 2026-08-26; unlike the native
bootstrap revision, that observation is not yet enforced by SDK resolution.
It contains ranks 0–1 and two parts per rank. The original archive lacks the
later `NOTICE` and per-shard checksum manifest present in the extracted Hugging
Face model directory; this historical limitation is not hidden or repaired by
changing archive bytes. The HF directory retains the Qwen Research License,
modification notice and recorded shard checksums.

The SDK's Apache-2.0 license does not apply to Qwen weights. Qwen materials are
non-commercial research/evaluation materials under their included agreement;
commercial use requires a separate license from Alibaba Cloud.

## Evidence lineage

The Git-safe curated 2026-08-23 evidence is under `artifacts/`. Executed
notebooks remain under `kaggle-notebooks/`. The 2026-08-30 development-candidate
acceptance identifies source `7327b0b0…`; the separate controlled benchmark
identifies `6d10912a…`. Output-free final-package and Qwen regression notebooks
are future execution inputs. Historical evidence is never rewritten to appear
newer.

## Small SDK reproducibility

Wheel ZIP timestamps and build-tool versions can change bytes even when source
is equivalent. Two independent clean builds with build 1.5.0 and
`SOURCE_DATE_EPOCH=1787760000` produced byte-identical release-candidate wheels.
Their extracted sdists had identical file contents but the compressed archives
differed: setuptools stamped generated `PKG-INFO`, `setup.cfg`, egg-info and
directory tar headers with build time, and gzip recorded that build-time mtime
instead of the fixed epoch. Therefore deterministic wheels are demonstrated
for this controlled setup; deterministic sdists and universal byte-for-byte
reproducibility are not claimed. Published 0.1.x bytes remain immutable.
