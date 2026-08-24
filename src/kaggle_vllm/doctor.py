"""Human-readable checks for the validated Kaggle dual-T4 profile."""

from __future__ import annotations

from pathlib import Path

from .environment import Environment, collect
from .runtime import all_sm75, all_tesla_t4

VALIDATED_PROFILE = {
    "python_major_minor": "3.12",
    "torch_prefix": "2.10.0",
    "torch_cuda": "12.8",
    "gpu_count": 2,
    "gpu_name": "Tesla T4",
    "gpu_capability": (7, 5),
    "nccl": "2.27.5",
}


def suggested_build_env() -> dict[str, str]:
    """Return build settings derived from the successful Kaggle experiment."""

    environment = {
        "CUDA_HOME": "/usr/local/cuda",
        "CUDAToolkit_ROOT": "/usr/local/cuda",
        "VLLM_TARGET_DEVICE": "cuda",
        "TORCH_CUDA_ARCH_LIST": "7.5",
        "MAX_JOBS": "1",
        "NVCC_THREADS": "1",
    }
    if Path("/usr/local/nvidia/lib64/libcuda.so").exists():
        environment["CMAKE_LIBRARY_PATH"] = "/usr/local/nvidia/lib64"
    return environment


def profile_failures(environment: Environment) -> list[str]:
    """Return differences from the exactly validated Kaggle runtime profile."""

    failures: list[str] = []
    if not environment.is_kaggle:
        failures.append("Kaggle runtime markers were not detected")
    if not environment.python.startswith(VALIDATED_PROFILE["python_major_minor"] + "."):
        failures.append("Python is not 3.12.x")
    if environment.torch is None or not environment.torch.startswith(
        VALIDATED_PROFILE["torch_prefix"]
    ):
        failures.append("PyTorch is not 2.10.0.x")
    if environment.torch_cuda != VALIDATED_PROFILE["torch_cuda"]:
        failures.append("PyTorch CUDA ABI is not 12.8")
    if not environment.cuda_available:
        failures.append("CUDA is unavailable to PyTorch")
    if environment.gpu_count != VALIDATED_PROFILE["gpu_count"]:
        failures.append("Expected exactly two visible GPUs")
    if not all_tesla_t4(environment.gpus):
        failures.append("Visible GPUs are not all Tesla T4")
    if not all_sm75(environment.gpus):
        failures.append("Visible GPUs are not all compute capability 7.5 / SM75")
    if environment.cuda_driver is None:
        failures.append("CUDA driver library was not found")
    return failures


def run_doctor(environment: Environment | None = None) -> int:
    """Print diagnostics and return zero only for the validated profile."""

    runtime = environment or collect()
    failures = profile_failures(runtime)
    print("kaggle-vllm doctor")
    print("==================")
    print(f"Kaggle       : {runtime.is_kaggle}")
    print(f"Python       : {runtime.python}")
    print(f"PyTorch      : {runtime.torch or 'NOT FOUND'}")
    print(f"PyTorch path : {runtime.torch_path or 'NOT FOUND'}")
    print(f"PyTorch CUDA : {runtime.torch_cuda or 'NOT FOUND'}")
    print(f"CUDA usable  : {runtime.cuda_available}")
    print(f"NCCL         : {runtime.nccl or 'NOT FOUND'}")
    print(f"nvcc         : {runtime.nvcc or 'NOT FOUND'}")
    print(f"CUDA driver  : {runtime.cuda_driver or 'NOT FOUND'}")
    print(f"GPUs         : {runtime.gpu_count}")
    for gpu in runtime.gpus:
        print(
            f"  GPU {gpu.index}: {gpu.name}, compute "
            f"{gpu.capability[0]}.{gpu.capability[1]}"
        )

    print("\nSuggested build environment:")
    for key, value in suggested_build_env().items():
        print(f"  export {key}={value}")

    if runtime.nccl is None:
        print("\nWarning: could not read the NCCL version")
    elif runtime.nccl != VALIDATED_PROFILE["nccl"]:
        print(
            f"\nWarning: NCCL {runtime.nccl} differs from the validated "
            f"{VALIDATED_PROFILE['nccl']}"
        )

    if failures:
        print("\nPROFILE MISMATCH:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("\nPASS: runtime matches the documented Kaggle T4x2 profile.")
    return 0
