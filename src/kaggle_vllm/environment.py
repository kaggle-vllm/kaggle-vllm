"""Side-effect-free runtime discovery for Kaggle and CUDA environments."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GPUInfo:
    """A visible CUDA device reported by PyTorch."""

    index: int
    name: str
    capability: tuple[int, int]
    total_memory: int | None = None

    @property
    def is_tesla_t4(self) -> bool:
        return "tesla t4" in self.name.casefold()

    @property
    def is_sm75(self) -> bool:
        return self.capability == (7, 5)


@dataclass(frozen=True)
class Environment:
    """A serializable snapshot of the local runtime."""

    is_kaggle: bool
    python: str
    platform: str
    torch: str | None
    torch_path: str | None
    torch_cuda: str | None
    cuda_available: bool
    gpus: tuple[GPUInfo, ...]
    nccl: str | None
    nvcc: str | None
    nvcc_version: str | None
    cuda_home: str | None
    cuda_driver: str | None
    cmake_library_path: str | None

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    @property
    def gpu_names(self) -> list[str]:
        return [gpu.name for gpu in self.gpus]

    @property
    def gpu_caps(self) -> list[tuple[int, int]]:
        return [gpu.capability for gpu in self.gpus]


def detect_kaggle(
    environ: Mapping[str, str] | None = None,
    kaggle_root: Path = Path("/kaggle"),
) -> bool:
    """Return whether common Kaggle runtime markers are present."""

    current = os.environ if environ is None else environ
    markers = ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_DOCKER_IMAGE")
    return any(current.get(marker) for marker in markers) or (
        (kaggle_root / "working").is_dir() and (kaggle_root / "input").is_dir()
    )


def _run(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def find_cuda_driver(candidates: tuple[Path, ...] | None = None) -> str | None:
    """Locate the live CUDA driver library without mutating the environment."""

    paths = candidates or (
        Path("/usr/local/nvidia/lib64/libcuda.so"),
        Path("/usr/local/nvidia/lib64/libcuda.so.1"),
        Path("/usr/local/cuda/compat/libcuda.so"),
    )
    return next((str(path) for path in paths if path.exists()), None)


def _gpu_memory(torch_module: Any, index: int) -> int | None:
    try:
        return int(torch_module.cuda.get_device_properties(index).total_memory)
    except (AttributeError, OSError, RuntimeError):
        return None


def collect(torch_module: Any | None = None) -> Environment:
    """Collect a runtime fingerprint; PyTorch is optional and lazily imported."""

    torch = torch_module
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except (ImportError, OSError, RuntimeError):
            torch = None

    if torch is None:
        torch_version = torch_path = torch_cuda = nccl = None
        cuda_available = False
        gpus: tuple[GPUInfo, ...] = ()
    else:
        torch_version = str(torch.__version__)
        torch_path = str(getattr(torch, "__file__", "")) or None
        torch_cuda = getattr(torch.version, "cuda", None)
        try:
            cuda_available = bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count())
            gpus = tuple(
                GPUInfo(
                    index=index,
                    name=str(torch.cuda.get_device_name(index)),
                    capability=tuple(torch.cuda.get_device_capability(index)),
                    total_memory=_gpu_memory(torch, index),
                )
                for index in range(count)
            )
        except (AssertionError, AttributeError, OSError, RuntimeError):
            cuda_available = False
            gpus = ()
        try:
            version = torch.cuda.nccl.version()
            nccl = ".".join(map(str, version))
        except (AttributeError, OSError, RuntimeError, TypeError):
            nccl = None

    nvcc = shutil.which("nvcc")
    return Environment(
        is_kaggle=detect_kaggle(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch_version,
        torch_path=torch_path,
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        gpus=gpus,
        nccl=nccl,
        nvcc=nvcc,
        nvcc_version=_run(nvcc, "--version") if nvcc else None,
        cuda_home=os.environ.get("CUDA_HOME"),
        cuda_driver=find_cuda_driver(),
        cmake_library_path=os.environ.get("CMAKE_LIBRARY_PATH"),
    )


def as_json(environment: Environment | None = None) -> str:
    """Serialize a runtime fingerprint without including environment secrets."""

    return json.dumps(asdict(environment or collect()), indent=2)
