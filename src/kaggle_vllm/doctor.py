from __future__ import annotations

import os
from pathlib import Path

from .environment import collect


TARGET = {
    "python_major_minor": "3.12",
    "torch_prefix": "2.10.0",
    "torch_cuda": "12.8",
    "gpu_count": 2,
    "gpu_name": "Tesla T4",
    "gpu_cap": (7, 5),
}


def suggested_build_env() -> dict[str, str]:
    env = {
        "CUDA_HOME": "/usr/local/cuda",
        "CUDAToolkit_ROOT": "/usr/local/cuda",
        "VLLM_TARGET_DEVICE": "cuda",
        "TORCH_CUDA_ARCH_LIST": "7.5",
        "MAX_JOBS": "1",
        "NVCC_THREADS": "1",
    }
    if Path("/usr/local/nvidia/lib64/libcuda.so").exists():
        env["CMAKE_LIBRARY_PATH"] = "/usr/local/nvidia/lib64"
    return env


def run_doctor() -> int:
    env = collect()
    failures: list[str] = []
    warnings: list[str] = []

    print("kaggle-vllm doctor")
    print("==================")
    print(f"Python       : {env.python}")
    print(f"PyTorch      : {env.torch}")
    print(f"PyTorch CUDA : {env.torch_cuda}")
    print(f"CUDA usable  : {env.cuda_available}")
    print(f"NCCL         : {env.nccl}")
    print(f"nvcc         : {env.nvcc or 'NOT FOUND'}")
    print(f"CUDA driver  : {env.cuda_driver or 'NOT FOUND'}")
    print(f"GPUs         : {env.gpu_count}")

    for i, (name, cap) in enumerate(zip(env.gpu_names, env.gpu_caps)):
        print(f"  GPU {i}: {name}, compute {cap[0]}.{cap[1]}")

    if not env.python.startswith(TARGET["python_major_minor"] + "."):
        failures.append("Python is not 3.12.x")
    if env.torch is None or not env.torch.startswith(TARGET["torch_prefix"]):
        failures.append("PyTorch is not 2.10.0.x")
    if env.torch_cuda != TARGET["torch_cuda"]:
        failures.append("PyTorch CUDA ABI is not 12.8")
    if not env.cuda_available:
        failures.append("CUDA is unavailable to PyTorch")
    if env.gpu_count != TARGET["gpu_count"]:
        failures.append("Expected exactly two GPUs")
    if any(n != TARGET["gpu_name"] for n in env.gpu_names):
        failures.append("Expected two Tesla T4 GPUs")
    if any(c != TARGET["gpu_cap"] for c in env.gpu_caps):
        failures.append("Expected compute capability 7.5 on both GPUs")
    if env.cuda_driver is None:
        failures.append("CUDA driver library was not found")
    if env.nccl is None:
        warnings.append("Could not read NCCL version")

    print("\nSuggested build environment:")
    for key, value in suggested_build_env().items():
        print(f"  export {key}={value}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(" -", w)

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(" -", f)
        return 1

    print("\nPASS: environment matches the initial Kaggle T4x2 compatibility profile.")
    return 0
