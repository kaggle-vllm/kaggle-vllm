# Local Python 3.11 checks (historical and current)

## Current source candidate

- `PYTHONPATH=src python3.11 -m pytest -q`: **428 passed, 1 skipped** after the historical native reference and HTTPS redirect checks.
- `python3.11 -m compileall -q src tests`: PASS.
- `python3.11 -m pytest -q tests/test_public_release_v032.py`: **9 passed** after the last downloader change.
- Two `SOURCE_DATE_EPOCH=1792152973 python3.11 -m build --no-isolation` builds produced byte-identical wheels and normalized sdists.
- `python3.11 -m twine check dist/kaggle_vllm-0.3.2*`: PASS for both artifacts.
- `sha256sum -c release/v032-SHA256SUMS.txt` from `dist/`: PASS. The manifest remains provisional because the immutable Hub revision is unset.
- Wheel metadata contains version 0.3.2, the `kaggle-vllm = kaggle_vllm.cli_v3:main` console entry point, both profiles and the 168-wheel source map. It contains no cp312 native wheel.
- Both acceptance notebooks passed JSON and Python syntax validation. Neither notebook nor any GPU runtime was executed.

Earlier build hashes below are historical snapshots and do not identify the current `dist/` files.

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

## Follow-up verification, 2026-10-16

After adding publisher-source delivery, `PYTHONPATH=src python3.11 -m pytest -q` passed **426 tests, 1 skipped**. The source-map test covers all 168 locked names, sizes and hashes and the non-PyPI Triton source. A separate read-only live smoke check downloaded the 9,352-byte `agent_detector` wheel from its pinned PyPI URL and verified its SHA256. The two primary acceptance notebooks passed JSON/Python AST structural checks only. No Kaggle notebook or GPU test was executed. Final distribution build and checksum results are recorded separately after build.

## Rebuilt source candidate

`python3.11 -m compileall -q src tests scripts` passed. Two Python 3.11 `build --no-isolation` runs with `SOURCE_DATE_EPOCH=1792152973` produced byte-identical wheels and normalized sdists. Twine passed both. The candidate wheel is `bfed3b92bdc56771f4501e3ccb3bc183abd8cf6498633d9fa5ce7e7f173c12e9`; the candidate sdist is `dbb3262738c2fae24282b5ce5d6cd5f3796a67c7c8a9d919c8f9e38739454523`. The pure Python wheel contains the locked 168-source map and no native wheel; its profile deliberately has no immutable Hub revision. These checksums must change after final Hub pinning.

A separate read-only remote check obtained HTTP 200 and expected Content-Length for all 168 publisher URLs. No Kaggle notebook or GPU runtime was launched.
