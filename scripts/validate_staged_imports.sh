#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/kaggle/working/vllm-staged}"
OVERLAY="${2:-}"

if [[ -n "$OVERLAY" ]]; then
  export PYTHONPATH="$OVERLAY:$TARGET${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}"
fi

TORCH_LIB="$(python3 - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python - <<'PY'
import importlib
import os
import sys

print("PYTHONPATH[0]:", sys.path[0])
print("staged target present:", "/kaggle/working/vllm-staged" in sys.path or any("vllm-staged" in p for p in sys.path))

mods = ["vllm", "vllm._C", "vllm._moe_C", "vllm.cumem_allocator"]
for name in mods:
    print(f"\nImporting {name} ...")
    mod = importlib.import_module(name)
    print("PASS:", getattr(mod, "__file__", mod))
PY
