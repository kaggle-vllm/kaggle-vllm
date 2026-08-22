#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/kaggle/working/vllm-staged}"

export PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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
