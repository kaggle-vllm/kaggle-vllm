#!/usr/bin/env bash
set -uo pipefail

SRC="${1:-/kaggle/working/vllm-0.18.1}"
WHEEL_DIR="${2:-/kaggle/working/kaggle-vllm-wheels}"
LOG="${3:-/kaggle/working/vllm-build.log}"

cd "$SRC"
mkdir -p "$WHEEL_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

# Kaggle-specific driver-library discovery.
if [[ -f /usr/local/nvidia/lib64/libcuda.so ]]; then
  export CMAKE_LIBRARY_PATH="/usr/local/nvidia/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
fi

export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
export MAX_JOBS="${MAX_JOBS:-1}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

python -m pip install --no-cache-dir \
  "setuptools-scm>=8" \
  "packaging>=24.2" \
  "jinja2>=3.1.6"

python -m pip wheel \
  --verbose \
  --no-build-isolation \
  --no-deps \
  --wheel-dir "$WHEEL_DIR" \
  . 2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
echo "BUILD EXIT STATUS: $status"
echo "LOG: $LOG"
exit "$status"
