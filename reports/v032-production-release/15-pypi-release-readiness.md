# PyPI 0.3.2 readiness

The package name, version, pure Python wheel, source distribution and CLI entry point have local checks. PyPI currently lists 0.3.1 as its latest release; 0.3.2 is not uploaded. The current 0.3.2 build intentionally has an unset Hub revision and cannot deliver a public runtime, so it is a release candidate rather than the final PyPI artifact. Rebuild after the complete immutable Hub pin is committed; verify wheel/sdist byte identity and trusted publishing configuration. Never reuse the previously disclosed token. A PyPI upload requires explicit user authorization.

The proposed `.github/workflows/publish-pypi-v032.yml` is guarded by the reviewed tag, the redistribution clearance file, exact HF completeness, fixed-epoch artifact checksums and a PyPI no-collision check. PyPI's Trusted Publisher configuration must explicitly trust that workflow filename and `pypi` environment before dispatch.

## Revised distribution gate

PyPI JSON returned 404 for 0.3.2 on 2026-10-16. The Trusted Publishing workflow now requires the exact 168-source publisher map and native redistribution clearance, then checks three immutable Hub entries (lock, source map, native) instead of an unapproved third-party wheel mirror. Its legal gate remains BLOCKED. A final SDK wheel/sdist and release checksums cannot be called final until the verified Hub commit is pinned. No PyPI upload was attempted.
