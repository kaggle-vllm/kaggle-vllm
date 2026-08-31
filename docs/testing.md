# Testing and acceptance

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
arguments and sharded topology. The integration GPU test remains marked and
skipped outside the exact environment.

## Kaggle GPU acceptance

Only a real Kaggle T4x2 run can validate native imports, Torch preservation,
NCCL, model initialization/generation, attention selection, sharded reload,
serving and performance. The 2026-08-30 development candidate passed its
acceptance and benchmark runs. The exact public `0.2.0` package and focused
existing-Qwen TP=2 regression then passed on 2026-08-31; see the
[final acceptance report](kaggle-v0.2.0-final-acceptance.md).

Development-candidate evidence and final published-package checks are reported
separately. A skipped local test, valid notebook JSON or successful CPU package
build is never a GPU pass.
