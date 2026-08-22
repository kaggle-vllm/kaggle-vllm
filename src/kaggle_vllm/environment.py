from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Environment:
    python: str
    platform: str
    torch: str | None
    torch_cuda: str | None
    cuda_available: bool
    gpu_count: int
    gpu_names: list[str]
    gpu_caps: list[tuple[int, int]]
    nccl: str | None
    nvcc: str | None
    nvcc_version: str | None
    cuda_home: str | None
    cuda_driver: str | None
    cmake_library_path: str | None


def _run(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception:
        return None


def _find_driver() -> str | None:
    candidates = [
        Path("/usr/local/nvidia/lib64/libcuda.so"),
        Path("/usr/local/nvidia/lib64/libcuda.so.1"),
        Path("/usr/local/cuda/compat/libcuda.so"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def collect() -> Environment:
    try:
        import torch
        torch_version = torch.__version__
        torch_cuda = torch.version.cuda
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        gpu_caps = [torch.cuda.get_device_capability(i) for i in range(gpu_count)]
        try:
            nccl = ".".join(map(str, torch.cuda.nccl.version()))
        except Exception:
            nccl = None
    except Exception:
        torch_version = None
        torch_cuda = None
        cuda_available = False
        gpu_count = 0
        gpu_names = []
        gpu_caps = []
        nccl = None

    nvcc = shutil.which("nvcc")
    return Environment(
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        gpu_count=gpu_count,
        gpu_names=gpu_names,
        gpu_caps=gpu_caps,
        nccl=nccl,
        nvcc=nvcc,
        nvcc_version=_run(nvcc, "--version") if nvcc else None,
        cuda_home=os.environ.get("CUDA_HOME"),
        cuda_driver=_find_driver(),
        cmake_library_path=os.environ.get("CMAKE_LIBRARY_PATH"),
    )


def as_json() -> str:
    return json.dumps(asdict(collect()), indent=2)
