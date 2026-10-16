"""Strict, packaged L4x4 runtime identity for SDK 0.3.2."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

from .exceptions import ProfileError

PROFILE_NAME = "kaggle-l4x4-cu129"
WHEEL_SHA256 = "9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8"
WHEEL_FILENAME = "vllm-0.30.0-cp312-cp312-linux_x86_64.whl"
NATIVE_HF_PATH = "v0.3.0/l4x4/native/" + WHEEL_FILENAME


@dataclass(frozen=True)
class L4Profile:
    name: str
    status: str
    python_abi: str
    gpu_name: str
    gpu_count: int
    compute_capability: tuple[int, int]
    tensor_parallel_size: int
    torch_version: str
    torch_cuda: str
    system_cuda_compiler: str
    vllm_version: str
    wheel_filename: str
    wheel_sha256: str
    data: dict[str, Any]


def validate_profile(data: dict[str, Any]) -> L4Profile:
    """Reject malformed or diluted identities, including an accidental T4 profile."""
    try:
        p, g, r, v = data["python"], data["gpu"], data["runtime"], data["vllm"]
        if data["schema_version"] != 3 or data["name"] != PROFILE_NAME or data["sdk_generation"] != "0.3.2":
            raise ValueError("wrong schema or profile name")
        if (p["implementation"], p["major"], p["minor"], p["abi"]) != ("CPython", 3, 12, "cp312"):
            raise ValueError("native Python ABI must be CPython cp312")
        if (data["platform"]["system"], data["platform"]["machine"]) != ("Linux", "x86_64"):
            raise ValueError("platform must be Linux x86_64")
        if (g["name"], g["count"], g["compute_capability"], g["architecture"], g["tensor_parallel_size"]) != ("NVIDIA L4", 4, [8, 9], "SM89", 4):
            raise ValueError("GPU profile must be exactly four NVIDIA L4 / SM89 with TP=4")
        if (r["torch"], r["torch_cuda"], r["system_cuda_compiler"], r["torchcodec"], r["nccl"]) != ("2.13.0+cu129", "12.9", "12.8", "0.16.0+cu129", "2.29.7"):
            raise ValueError("Torch/CUDA/TorchCodec identity mismatch")
        if (r["torchvision"], r["torchaudio"], r["triton"], r["flashinfer_python"], r["flashinfer_cubin"]) != ("0.28.0+cu129", "2.11.0+cu129", "3.7.1", "0.6.18.post1", "0.6.18.post1"):
            raise ValueError("runtime component identity mismatch")
        if (v["version"], v["source_tag"], v["source_commit"], v["wheel_filename"], v["wheel_sha256"]) != ("0.30.0", "v0.30.0", "ced6857afa0ea7b2e3f0846a62e1394e90f15607", WHEEL_FILENAME, WHEEL_SHA256):
            raise ValueError("vLLM native wheel identity mismatch")
        if not re.fullmatch(r"[0-9a-f]{64}", v["wheel_sha256"]):
            raise ValueError("invalid SHA256")
        if v["wheel_size"] != 293855397 or data["default_runtime_root"] != "/tmp/kaggle-vllm/v0.3.2/l4x4":
            raise ValueError("wheel size or scratch runtime root mismatch")
        if data["public_artifacts"]["hf_repo_id"] != "waqasm86/kaggle-vllm-binaries":
            raise ValueError("wrong public binary authority")
        if data["public_artifacts"].get("namespace") != "v0.3.2/l4x4/":
            raise ValueError("wrong public runtime namespace")
        if data["public_artifacts"].get("native_remote_path") != NATIVE_HF_PATH:
            raise ValueError("wrong immutable native wheel path")
        if data["public_artifacts"].get("wheelhouse_delivery") != "publisher-index":
            raise ValueError("wrong locked wheel delivery mode")
        if v["gemma4_tested_backend"] != "TRITON_ATTN":
            raise ValueError("Gemma 4 backend evidence mismatch")
        evidence = data["evidence"]
        if (evidence["runtime_v3_sha256"], evidence["runtime_v3_size"], evidence["runtime_manifest_sha256"]) != ("0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73", 6633730070, "3fbe937af3cc281c4dd1eeadcf1c35fade1c4c2162fca7ac116a942f674a217a"):
            raise ValueError("runtime-v3 identity mismatch")
        return L4Profile(data["name"], data["status"], p["abi"], g["name"], g["count"], tuple(g["compute_capability"]), g["tensor_parallel_size"], r["torch"], r["torch_cuda"], r["system_cuda_compiler"], v["version"], v["wheel_filename"], v["wheel_sha256"], data)
    except (KeyError, TypeError, ValueError) as error:
        raise ProfileError(f"invalid L4x4 profile: {error}") from error


def load_profile(profile_name: str = PROFILE_NAME) -> L4Profile:
    if profile_name != PROFILE_NAME:
        raise ProfileError(f"0.3.2 supports only {PROFILE_NAME}")
    try:
        payload = json.loads(resources.files("kaggle_vllm").joinpath("profiles", PROFILE_NAME, "profile.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"cannot read profile: {error}") from error
    return validate_profile(payload)
