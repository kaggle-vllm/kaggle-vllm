#!/usr/bin/env bash
set -euo pipefail

WHEEL="${1:?usage: stage_wheel.sh /path/to/vllm-*.whl [target]}"
TARGET="${2:-/kaggle/working/vllm-staged}"

rm -rf "$TARGET"
mkdir -p "$TARGET"

echo "Staging wheel WITHOUT dependency resolution:"
echo "  wheel : $WHEEL"
echo "  target: $TARGET"

python -m pip install \
  --no-deps \
  --target "$TARGET" \
  "$WHEEL"

echo
echo "Staged. Kaggle's normal site-packages were not used as the install target."
