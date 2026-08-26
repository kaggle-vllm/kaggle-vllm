# Security and supply chain

## Trust boundary

Bootstrap accepts only the artifact identity packaged in a reviewed profile:
immutable Hugging Face commit, exact filename and exact SHA256. The download is
streamed to a temporary file, verified and then moved into cache. Staging uses
argument arrays and `pip --target --no-deps`; serving also uses an argument
array and never `shell=True`.

Imports have no network, download, installation or activation side effects.
Runtime activation is explicit and does not edit persistent shell files.

## Filesystem ownership

Bootstrap refuses non-empty destinations by default. Reset requires
`--reset-runtime --yes`, rejects root/system/Kaggle parent paths, current
repository paths, overlaps and symlink traversal, and removes only known
defaults or paths proved by a matching manifest. The verified download cache
is preserved by default.

Sharded-state inspection rejects symlinked roots/members so an apparent shard
cannot escape the inspected directory. The SDK currently does not extract
model tar archives. Any future extractor must reject absolute/`..` member
paths, links, devices and other unsafe member types before writing.

## Operational exposure

The server defaults to `127.0.0.1`. Binding `0.0.0.0` exposes the process on all
available interfaces and requires authentication, authorization, TLS and
network controls outside this SDK. No tunnel or public exposure is created
automatically.

## Reporting vulnerabilities

Do not open a public issue for a credible vulnerability. Follow
[`SECURITY.md`](../SECURITY.md). Never include tokens, cookies or credentials in
notebooks, manifests, logs or issue attachments.

## Release verification

Run `scripts/verify_static.py`, checksum large artifacts from their
authoritative read-only locations and inspect wheel/sdist contents. Generate a
CycloneDX SBOM for the small SDK when desired:

```bash
python -m pip install cyclonedx-bom
cyclonedx-py environment --output-file sdk-sbom.cdx.json
```

Review generated SBOM scope before publication; an environment SBOM is not the
same as an SBOM for the separately staged native runtime.
