# Kaggle runtime

The validated runtime was captured on 2026-08-22 and exercised through
2026-08-23. Exact values are in `compat/kaggle-t4x2-cu128.json` and the curated
environment manifest under `artifacts/`.

## Diagnostics

```bash
kaggle-vllm fingerprint
kaggle-vllm doctor
kaggle-vllm verify-gpus --tensor-parallel-size 2
```

`fingerprint` is informational and works without vLLM. `doctor` is strict: it
returns a non-zero result if the runtime differs from the exact tested profile.
That does not assert every other environment is broken; it means it is outside
the evidence boundary.

## T4 behavior

Tesla T4 is compute capability 7.5 (SM75). The logs show FlashAttention 2 being
rejected because it requires compute capability 8.0 or newer, followed by vLLM
selecting `TRITON_ATTN`. The two SymmMem communicator warnings are also expected
on SM75. Neither warning prevented ordinary NCCL TP=2 communication.

## Driver and toolkit paths

The CUDA 12.8 toolkit was under `/usr/local/cuda-12.8` (with the usual
`/usr/local/cuda` path), but Kaggle mounted the driver at
`/usr/local/nvidia/lib64/libcuda.so`. Setting
`CMAKE_LIBRARY_PATH=/usr/local/nvidia/lib64` allowed CMake to create
`CUDA::cuda_driver` during the successful source build.

`nvidia-smi` reported a driver CUDA maximum compatibility of 13.0. This is
distinct from both the CUDA 12.8 toolkit/PyTorch `cu128` ABI and the T4 GPU's
compute capability 7.5 / SM75.
