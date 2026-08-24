#!/usr/bin/env bash
set -euo pipefail

WHEEL="${1:?usage: stage_wheel.sh /path/to/vllm-*.whl [target]}"
TARGET="${2:-/kaggle/working/vllm-staged}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m kaggle_vllm.cli stage-wheel "$WHEEL" --target "$TARGET"
