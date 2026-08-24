# Installation and recovery

## Why staging exists

Kaggle ships a coordinated PyTorch/CUDA/Triton environment. A normal vLLM pip
install can resolve a large dependency graph and replace packages in that
environment. The validated recovery kept PyTorch 2.10.0+cu128 at its system
path, staged only the wheel with `--no-deps`, and created a separate pinned
overlay for missing Python dependencies.

## Wheel integrity and staging

```bash
DIGEST=5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c
kaggle-vllm verify-wheel /kaggle/input/.../vllm-*.whl --sha256 "$DIGEST"
kaggle-vllm stage-wheel /kaggle/input/.../vllm-*.whl \
  --target /kaggle/working/vllm-staged \
  --sha256 "$DIGEST"
```

The destination must be absent or empty. The tool does not delete an existing
environment and does not perform a global install.

## Dependency overlay

The archived `vllm-overlay-lock-v3.txt` is the reproducible input from the
successful recovery. Python callers can use `stage_dependency_overlay()`; it
rejects `torch`, `torchvision`, and `torchaudio` requirement entries, uses
`pip --target --no-deps`, and requires an empty target.

The runtime ordering proven in the notebook was:

```text
PYTHONPATH=/kaggle/working/vllm-runtime-overlay:/kaggle/working/vllm-staged
```

`LD_LIBRARY_PATH` included the existing Torch library directory, the mounted
driver directory, and the CUDA toolkit library directory. Validate native
imports before running inference.

## Source build

`scripts/prepare_vllm_0181.sh` checks out the exact upstream tag/commit and runs
vLLM's `use_existing_torch.py --prefix`. `scripts/build_vllm_0181.sh` selects
CUDA/SM75, applies the driver discovery path, avoids dependency wheel building,
and records a full log. These scripts are Kaggle-oriented and are not invoked by
SDK import.
