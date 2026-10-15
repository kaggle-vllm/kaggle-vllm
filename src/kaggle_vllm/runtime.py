"""Hardware checks shared by inference, diagnostics, and the CLI."""

from __future__ import annotations

from .environment import Environment, GPUInfo, collect
from .exceptions import RuntimeValidationError


def visible_gpu_count(environment: Environment | None = None) -> int:
    """Return the number of CUDA devices visible to PyTorch."""

    return (environment or collect()).gpu_count


def all_tesla_t4(gpus: tuple[GPUInfo, ...] | list[GPUInfo]) -> bool:
    """Return true when the non-empty device collection contains only T4s."""

    return bool(gpus) and all(gpu.is_tesla_t4 for gpu in gpus)


def all_sm75(gpus: tuple[GPUInfo, ...] | list[GPUInfo]) -> bool:
    """Return true when the non-empty device collection contains only SM75 GPUs."""

    return bool(gpus) and all(gpu.is_sm75 for gpu in gpus)


def all_nvidia_l4(gpus: tuple[GPUInfo, ...] | list[GPUInfo]) -> bool:
    return bool(gpus) and all(gpu.is_nvidia_l4 for gpu in gpus)


def all_sm89(gpus: tuple[GPUInfo, ...] | list[GPUInfo]) -> bool:
    return bool(gpus) and all(gpu.is_sm89 for gpu in gpus)


def validate_l4x4_profile(environment: Environment | None = None) -> None:
    """Reject every topology except four exact NVIDIA L4 / SM89 devices."""
    snapshot = environment or collect()
    if not getattr(snapshot, "cuda_available", True) or snapshot.gpu_count != 4 or not all_nvidia_l4(snapshot.gpus) or not all_sm89(snapshot.gpus):
        raise RuntimeValidationError(
            "kaggle-vllm 0.3.0 requires exactly 4 NVIDIA L4 GPUs at SM89"
        )


def validate_tensor_parallel_size(
    tensor_parallel_size: int,
    gpu_count: int | None = None,
) -> None:
    """Validate a requested local TP degree against visible CUDA devices."""

    if (
        not isinstance(tensor_parallel_size, int)
        or isinstance(tensor_parallel_size, bool)
        or tensor_parallel_size < 1
    ):
        raise RuntimeValidationError("tensor_parallel_size must be a positive integer")
    visible = visible_gpu_count() if gpu_count is None else gpu_count
    if tensor_parallel_size > visible:
        raise RuntimeValidationError(
            f"tensor_parallel_size={tensor_parallel_size} requires at least "
            f"{tensor_parallel_size} visible GPUs; found {visible}"
        )
