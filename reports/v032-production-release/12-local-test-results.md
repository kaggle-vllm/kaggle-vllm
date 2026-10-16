# Local Python 3.11 checks

- `PYTHONPATH=src python3.11 -m pytest -q`: 425 passed, 1 skipped (final local CPU suite).
- `python3.11 -m compileall -q src tests`: PASS.
- `python3.11 -m build --no-isolation`: built the SDK wheel and source distribution without creating an environment.
- `python3.11 -m twine check dist/kaggle_vllm-0.3.2*`: PASS.
- Two fixed-epoch wheel builds matched byte-for-byte. Raw setuptools sdists differed in generated timestamps; normalized sdists matched byte-for-byte after `scripts/normalize_sdist_v032.py`. This is a local reproducibility check, not a claim that a later source commit will have the same checksum.
- Wheel metadata: version 0.3.2, Python `>=3.11,<3.13`, pure Python tag, CLI entry point and both profiles present. Sdist contains the new network and downloader modules.
- The exact 0.2.0 PyPI wheel passed Python 3.11 import/profile loading from the wheel archive without installing it.
- Distribution SHA256 values are in `release/v032-SHA256SUMS.txt`.

The first pytest invocation without `PYTHONPATH=src` loaded an older editable install from the dirty main checkout; its 10 failures are an environment path collision, not this worktree. All final tests target this worktree explicitly. Neither cp312 native wheel nor any Kaggle notebook was run locally.

Draft PR #39 initial checks passed: Python 3.11 CPU SDK, Python 3.12 CPU SDK, and quality-and-package. These CI jobs did not run Kaggle GPU acceptance.
