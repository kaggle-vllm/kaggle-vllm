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
