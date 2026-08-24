#!/usr/bin/env bash
set -euo pipefail

WHEEL="${1:?usage: inspect_wheel.sh /path/to/vllm-*.whl}"
DEST="${2:-/kaggle/working/vllm-wheel-inspect}"

if [[ -e "$DEST" ]]; then
  echo "Refusing to overwrite existing inspection path: $DEST" >&2
  exit 1
fi
mkdir -p "$DEST"

python - "$WHEEL" "$DEST" <<'PY'
import sys, zipfile
wheel, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(wheel) as z:
    z.extractall(dest)
    print("Shared libraries:")
    for n in z.namelist():
        if n.endswith(".so"):
            print(" ", n)
PY

echo
echo "ELF dependencies:"
while IFS= read -r -d '' so; do
  echo "---- $so"
  ldd "$so" || true
done < <(find "$DEST/vllm" -type f -name '*.so' -print0)

if command -v cuobjdump >/dev/null 2>&1; then
  echo
  echo "CUDA ELF targets:"
  while IFS= read -r -d '' so; do
    echo "---- $so"
    cuobjdump --list-elf "$so" 2>/dev/null | grep -E 'sm_[0-9]+' | sort -u || true
  done < <(find "$DEST/vllm" -type f -name '*.so' -print0)
fi
