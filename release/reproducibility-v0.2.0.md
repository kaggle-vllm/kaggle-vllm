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
reproducibility results, not published-artifact identities. After trusted
publishing, download the exact PyPI wheel and sdist and only then append their
verified hashes to `kaggle-vllm-sdk-SHA256SUMS.txt`; the published sdist is
expected to differ from the local candidate because of the timestamp behavior
above.
