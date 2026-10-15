# Testing and acceptance

For the v0.3.0 local candidate, run `python3.11 -m pytest -q`, `python3.11 -m build --no-isolation`, and `python3.11 -m twine check` with the two exact 0.3.0 distribution paths. Python 3.11 tests mock the cp312/GPU boundaries. They do not import or rebuild the CUDA wheel. The [final L4 SDK notebook](../kaggle-notebooks/kaggle_vllm_0_3_0_l4x4_final_sdk_acceptance.ipynb) is a separate fresh CPython 3.12/four-L4 gate. Its evidence status is `FRESH_KAGGLE_PENDING` until run; local tests never imply a GPU pass.

## CPU SDK CI

GitHub CI tests the lightweight SDK on Python 3.10, 3.11, 3.12 and 3.13. It
runs unit tests, import/CLI/bootstrap-plan smokes, Ruff, profile/provenance
consistency, Markdown link checks, package build, `twine check` and distribution
content inspection. It neither installs the native wheel nor pretends to test
CUDA on a CPU runner.

```bash
python -m pip install -e ".[test]"
pytest -q
ruff check src tests examples scripts
python scripts/verify_static.py
python scripts/check_markdown_links.py
python -m build
python -m twine check dist/*
```

Unit tests cover profile parsing, dependency absence/range/drift/malformed
metadata, dry-run planning, immutable definitions, checksums, reset ownership,
path overlap/symlinks, no implicit bootstrap, TP validation, safe server
arguments, sharded topology, benchmark configuration/schema/statistics,
comparison deltas, topology parsing, telemetry failure handling and evidence
output safety. The integration GPU test remains marked and skipped outside the
exact environment.

## Kaggle GPU acceptance

Only a real Kaggle T4x2 run can validate native imports, Torch preservation,
NCCL, model initialization/generation, attention selection, sharded reload,
serving and performance. The 2026-08-30 development candidate passed its
acceptance and benchmark runs. The exact public `0.2.0` package and focused
existing-Qwen TP=2 regression then passed on 2026-08-31; see the
[final acceptance report](kaggle-v0.2.0-final-acceptance.md).

The post-0.2.0 Milestone 1 benchmark implementation is locally testable, but
its new matrix remains **unexecuted** until the output-free Kaggle launcher is
run on the real profile. A dry-run plan, mocked engine output or topology
parser fixture is never a GPU performance result.

Development-candidate evidence and final published-package checks are reported
separately. A skipped local test, valid notebook JSON or successful CPU package
build is never a GPU pass.
