#!/usr/bin/env bash
set -euo pipefail

TEST="${1:-/kaggle/working/cmake-cuda-driver-probe}"
if [[ -e "$TEST" ]]; then
  echo "Refusing to overwrite existing probe path: $TEST" >&2
  exit 1
fi
mkdir -p "$TEST"

cat > "$TEST/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.26)
project(cuda_driver_test LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

if(TARGET CUDA::cuda_driver)
  get_target_property(CUDA_DRIVER_LOCATION CUDA::cuda_driver IMPORTED_LOCATION)
  message(STATUS "SUCCESS: CUDA::cuda_driver=${CUDA_DRIVER_LOCATION}")
else()
  message(FATAL_ERROR "CUDA::cuda_driver does NOT exist")
endif()
EOF

export CMAKE_LIBRARY_PATH="/usr/local/nvidia/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"

cmake \
  -S "$TEST" \
  -B "$TEST/build" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_LIBRARY_PATH="$CMAKE_LIBRARY_PATH"
