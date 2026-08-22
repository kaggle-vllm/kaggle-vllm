# Build findings

## Proven

The Kaggle image used for the successful build supplied Python 3.12.13,
PyTorch 2.10.0+cu128, CUDA 12.8.93, NCCL 2.27.5, CMake 3.31.10, GCC 11.4,
and two Tesla T4 GPUs with compute capability 7.5.

The source checkout was upstream vLLM tag `v0.18.1`, commit:

`a26e8dc7ff2111a005144d775ecf9cebf56c45b2`

The build completed and produced the experimental wheel recorded in
`compat/kaggle-t4x2-cu128.json`.

## Kaggle CUDA-driver discovery issue

`find_package(CUDAToolkit REQUIRED)` could find CUDA 12.8 but failed to expose
`CUDA::cuda_driver` until CMake was given:

`CMAKE_LIBRARY_PATH=/usr/local/nvidia/lib64`

The live driver library was:

`/usr/local/nvidia/lib64/libcuda.so`

## Architecture nuance

The primary vLLM CUDA build selected SM75 and produced explicit SM75 Marlin
translation units. The bundled vLLM FlashAttention external project also built
SM80 and SM90 objects. Therefore the first wheel should be described as a
Kaggle/T4-targeted build, not as a strictly SM75-only binary.

A later optimization pass can remove irrelevant FA3/Hopper code after runtime
validation is complete.
