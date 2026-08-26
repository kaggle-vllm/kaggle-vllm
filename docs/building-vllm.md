# Building upstream vLLM for Kaggle

Building is historical reproduction guidance, not the normal install path.
Use upstream vLLM tag `v0.18.1` at commit
`a26e8dc7ff2111a005144d775ecf9cebf56c45b2`; do not infer source v0.18.2 from
the generated distribution version.

The successful environment and conservative settings are captured by
`scripts/prepare_vllm_0181.sh`, `scripts/build_vllm_0181.sh` and
`kaggle-vllm build-env`. Important values include:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDAToolkit_ROOT=/usr/local/cuda
export VLLM_TARGET_DEVICE=cuda
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=1
export NVCC_THREADS=1
export CMAKE_LIBRARY_PATH=/usr/local/nvidia/lib64
```

## Kaggle driver discovery

Kaggle provided the CUDA 12.8 toolkit under `/usr/local/cuda-12.8`, while the
live driver mount was `/usr/local/nvidia/lib64/libcuda.so`. CMake could find
the toolkit yet fail to expose `CUDA::cuda_driver` until
`CMAKE_LIBRARY_PATH=/usr/local/nvidia/lib64` was set. This was a managed
platform filesystem/layout discovery issue—not evidence of a missing or broken
NVIDIA driver.

After building, record source identity, environment, build flags, wheel name,
SHA256, ZIP metadata, native extensions, ELF dependencies and CUDA
architectures. A new build must use a new artifact identity/revision; it must
not replace the historical wheel or its checksum.

See [provenance](provenance.md) and historical [build findings](BUILD_FINDINGS.md).
