#!/usr/bin/env bash
set -euo pipefail

VLLM_TAG="v0.18.1"
VLLM_COMMIT="a26e8dc7ff2111a005144d775ecf9cebf56c45b2"
ROOT="${1:-/kaggle/working/vllm-0.18.1}"

if [[ -e "$ROOT" ]]; then
  echo "Refusing to overwrite $ROOT"
  exit 1
fi

git clone --branch "$VLLM_TAG" --depth 1 \
  https://github.com/vllm-project/vllm.git "$ROOT"

cd "$ROOT"
actual="$(git rev-parse HEAD)"

if [[ "$actual" != "$VLLM_COMMIT" ]]; then
  echo "Unexpected commit: $actual"
  exit 1
fi

python use_existing_torch.py --prefix

echo "Prepared vLLM $VLLM_TAG at $actual"
