"""Compatibility launcher for ``kaggle-vllm benchmark``.

The original one-off 0.2.0 harness is preserved with its historical evidence
under ``artifacts/kaggle-2026-08-30-v0.2.0-benchmark``. New runs use the tested
SDK implementation so benchmark logic is not duplicated in this script.
"""

from __future__ import annotations

import sys

from kaggle_vllm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
