# vLLM Kaggle NVIDIA Dual T4 GPUs

Kaggle-specific build, compatibility, and validation toolkit for running vLLM on
**2× NVIDIA Tesla T4 (SM 7.5)** while preserving Kaggle's preinstalled
Python/PyTorch/CUDA environment.

> **Status:** development / research. The build is proven; runtime validation is
> intentionally staged and not yet declared production-ready.

## Proven build profile

A CUDA wheel was successfully built in Kaggle from upstream **vLLM v0.18.1**
(commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`) against:

- Ubuntu 22.04 / glibc 2.35
- Python 3.12.13
- PyTorch `2.10.0+cu128`
- CUDA toolkit 12.8.93
- NVIDIA driver 580.159.04
- NCCL 2.27.5
- CMake 3.31.10
- GCC 11.4.0
- 2× NVIDIA Tesla T4, compute capability 7.5

Successful experimental wheel:

`vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`

SHA256:

`5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

The wheel version string comes from `setuptools_scm`; the source checkout itself
is the upstream `v0.18.1` tag.

## What this project solves

Normal vLLM installation may resolve and replace tightly-coupled packages
already supplied by Kaggle. This project instead:

1. fingerprints the Kaggle runtime;
2. preserves Kaggle's existing PyTorch/CUDA/NCCL/Triton stack;
3. builds vLLM against the existing PyTorch installation;
4. targets T4/SM75 for the primary vLLM CUDA extensions;
5. fixes Kaggle-specific CUDA-driver discovery for CMake;
6. validates built wheels with `pip --target --no-deps` before normal install;
7. validates single-GPU inference before NCCL and dual-GPU tensor parallelism.

## Kaggle CMake finding

Kaggle exposes the CUDA toolkit at `/usr/local/cuda`, while the live NVIDIA
driver library is mounted at:

`/usr/local/nvidia/lib64/libcuda.so`

On the measured Kaggle image, `find_package(CUDAToolkit)` discovered CUDA 12.8
but did not create `CUDA::cuda_driver` until the driver path was made visible:

```bash
export CMAKE_LIBRARY_PATH=/usr/local/nvidia/lib64
```

This compatibility check is part of `kaggle-vllm doctor`.

## Quick development workflow

```bash
python -m pip install -e .
kaggle-vllm doctor
kaggle-vllm fingerprint
```

For a source build:

```bash
bash scripts/prepare_vllm_0181.sh
bash scripts/build_vllm_0181.sh
```

For **non-production staged validation** of an existing wheel:

```bash
bash scripts/stage_wheel.sh /path/to/vllm-*.whl
bash scripts/validate_staged_imports.sh
```

The staging workflow does not perform a normal vLLM installation into the
Kaggle environment.

## Initial v0.1 runtime scope

- FP16
- single T4 smoke test
- raw NCCL all-reduce
- tensor parallel size 2
- `disable_custom_all_reduce=True` initially
- `enforce_eager=True` initially
- no BF16 default on T4
- no FP8 KV cache
- no Conda, uv, or Docker requirement

## Binary distribution

The ~367 MiB wheel is intentionally excluded from Git history. Publish it later
as a GitHub Release asset, Kaggle Dataset, or other artifact store and always
ship the SHA256 and compatibility manifest with it.

## Upstream relationship

This repository is a Kaggle compatibility/build layer around upstream vLLM. It
is not a replacement for vLLM and should preserve upstream attribution and
third-party notices.
