# kaggle-vllm 0.2.0 SDK build reproducibility

Two independent clean source copies were built with Python 3.11, build 1.5.0,
isolated build environments, and `SOURCE_DATE_EPOCH=1787760000`.

## Wheel

Both builds produced a 36,732-byte
`kaggle_vllm-0.2.0-py3-none-any.whl` with SHA256:

`1d8d40f7aa59bd96e41d005982c2d315dc2d398929ffbc3c02682c3e13e387c1`

Result: **byte-for-byte reproducible** for this controlled build.

## Source distribution

The selected 43,283-byte `kaggle_vllm-0.2.0.tar.gz` has SHA256:

`05c0e3ea282205e58ea3c12dc1141a2368b9253c805e708df58d97dee838fe83`

The independent second build was 43,266 bytes with SHA256:

`29a7fe76b4c9791a9f374fd276ce38e1510771c139a41373d6851afde6a07cd0`

Result: **not byte-for-byte reproducible**. Extracted file contents were
identical. Setuptools used build-time mtimes for generated `PKG-INFO`,
`setup.cfg`, egg-info and directory tar headers, and gzip recorded build-time
mtimes `1788170452` and `1788170459` instead of the fixed epoch. No deterministic
sdist claim is made.

Both local candidate distributions passed `twine check`. These hashes are
reproducibility results, not published-artifact identities.

## Finalization re-check

A second two-directory build from merge commit
`020fca67ff197980886c3e725c5c60a6e1478c7c`, using the same
`SOURCE_DATE_EPOCH`, reproduced the wheel above exactly in both builds. Its two
sdists again had identical extracted trees but differed at the archive level:

- build 1: 43,095 bytes,
  `1d8afbcdaa7554a3ba69c13bfbcc36d35b565126394afd338f75a4ae105f5fc3`
- build 2: 43,122 bytes,
  `aa32bb22f0b96974a8509ee603ece203f44a313e2507c26794078389a76b6c6b`

This confirms the previously documented result rather than a deterministic
sdist claim.

## Exact published artifacts

The files downloaded from public PyPI after trusted publishing from source
commit `020fca67ff197980886c3e725c5c60a6e1478c7c` are:

- wheel: 36,732 bytes,
  `f3dce393c9e0bd43b9ba29a29ae14f9467857e5eea61390d41f512a52911fbbe`
- sdist: 43,104 bytes,
  `48ed97da07e54119939053e38f4e87900cf79316711d7b4558aed8122a66e3aa`

These published hashes—not the local candidate hashes—are the release-asset
identities recorded in `kaggle-vllm-sdk-SHA256SUMS.txt`.
