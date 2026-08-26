"""Human- and machine-readable checks for the validated Kaggle profile."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from .dependencies import DependencyFinding, inspect_dependencies
from .environment import Environment, collect
from .profiles import load_profile
from .runtime import all_sm75, all_tesla_t4


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

    profile = load_profile()
    failures: list[str] = []
    if not environment.is_kaggle:
        failures.append("Kaggle runtime markers were not detected")
    python_prefix = f"{profile.python_major}.{profile.python_minor}."
    if not environment.python.startswith(python_prefix):
        failures.append(
            f"Python is not {profile.python_major}.{profile.python_minor}.x"
        )
    if environment.torch != profile.torch_version:
        failures.append(f"PyTorch is not {profile.torch_version}")
    if environment.torch_cuda != profile.torch_cuda:
        failures.append(f"PyTorch CUDA ABI is not {profile.torch_cuda}")
    if not environment.cuda_available:
        failures.append("CUDA is unavailable to PyTorch")
    if environment.gpu_count != profile.gpu_count:
        failures.append(f"Expected exactly {profile.gpu_count} visible GPUs")
    if not all_tesla_t4(environment.gpus):
        failures.append("Visible GPUs are not all Tesla T4")
    if not all_sm75(environment.gpus):
        failures.append("Visible GPUs are not all compute capability 7.5 / SM75")
    if environment.cuda_driver is None:
        failures.append("CUDA driver library was not found")
    if environment.nccl != profile.nccl:
        failures.append(f"NCCL is not {profile.nccl}")
    return failures


def _doctor_payload(
    runtime: Environment,
    dependencies: tuple[DependencyFinding, ...],
) -> dict[str, object]:
    failures = profile_failures(runtime)
    counts = {
        status: sum(item.status == status for item in dependencies)
        for status in ("pass", "warning", "error", "untested")
    }
    return {
        "profile": "kaggle-t4x2-cu128",
        "environment": asdict(runtime),
        "profile_findings": [
            {"status": "error", "message": failure} for failure in failures
        ],
        "dependency_findings": [item.to_dict() for item in dependencies],
        "summary": {"dependencies": counts},
        "compatible": not failures and counts["error"] == 0,
    }


def run_doctor(
    environment: Environment | None = None,
    *,
    strict: bool = False,
    check_dependencies: bool = True,
    as_json_output: bool = False,
    version_lookup: Callable[[str], str] = metadata.version,
) -> int:
    """Print diagnostics and return zero only when required checks pass."""

    runtime = environment or collect()
    profile = load_profile()
    failures = profile_failures(runtime)
    dependencies = (
        inspect_dependencies(strict=strict, version_lookup=version_lookup)
        if check_dependencies
        else ()
    )
    payload = _doctor_payload(runtime, dependencies)
    if as_json_output:
        print(json.dumps(payload, indent=2))
        return 0 if payload["compatible"] else 1
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
    print(f"Driver       : {runtime.driver_version or 'NOT FOUND'}")
    print(
        "Driver CUDA  : "
        f"{runtime.driver_reported_cuda_max or 'NOT FOUND'} (reported maximum)"
    )
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
    elif runtime.nccl != profile.nccl:
        print(
            f"\nWarning: NCCL {runtime.nccl} differs from the validated {profile.nccl}"
        )

    if check_dependencies:
        print("\nDependency baseline:")
        for finding in dependencies:
            print(f"  {finding.status.upper():8} {finding.message}")

    dependency_errors = [item for item in dependencies if item.status == "error"]
    if failures or dependency_errors:
        print("\nPROFILE MISMATCH:")
        for failure in failures:
            print(" -", failure)
        if dependency_errors:
            print(" - dependency baseline has ERROR findings")
        return 1
    print("\nPASS: runtime matches the documented Kaggle T4x2 profile.")
    return 0
