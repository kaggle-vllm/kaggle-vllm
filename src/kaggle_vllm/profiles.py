"""Packaged, immutable native-runtime compatibility profiles."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

from .exceptions import ProfileError

DEFAULT_PROFILE = "kaggle-t4x2-cu128"
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class BootstrapProfile:
    """Target host topology and, when validated, native artifact identity."""

    name: str
    status: str
    python_version: str
    python_implementation: str
    python_major: int
    python_minor: int
    python_abi: str
    system: str
    machine: str
    torch_version: str
    torch_cuda: str
    cuda_toolkit: str | None
    gpu_name: str
    compute_capability: tuple[int, int]
    gpu_count: int
    nccl: str | None
    vllm_source_tag: str
    vllm_source_commit: str
    wheel_filename: str | None
    wheel_sha256: str | None
    hf_repo_id: str | None
    hf_revision: str | None
    overlay_lock: str | None
    overlay_requirements: str | None
    attention_backend: str
    wheel_status: str

    @property
    def native_ready(self) -> bool:
        return self.wheel_status == "validated" and all(
            (
                self.wheel_filename,
                self.wheel_sha256,
                self.hf_repo_id,
                self.hf_revision,
                self.overlay_lock,
            )
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> BootstrapProfile:
        """Validate and construct a profile from packaged JSON data."""

        try:
            capability = data["runtime"]["compute_capability"]
            if (
                not isinstance(capability, list)
                or len(capability) != 2
                or any(type(value) is not int for value in capability)
            ):
                raise ProfileError(
                    "profile compute capability must contain two integers"
                )
            profile = cls(
                name=str(data["name"]),
                status=str(data["status"]),
                python_version=str(data["python"]["target_version"]),
                python_implementation=str(data["python"]["implementation"]),
                python_major=int(data["python"]["major"]),
                python_minor=int(data["python"]["minor"]),
                python_abi=str(data["python"]["abi"]),
                system=str(data["platform"]["system"]),
                machine=str(data["platform"]["machine"]),
                torch_version=str(data["runtime"]["torch"]),
                torch_cuda=str(data["runtime"]["torch_cuda"]),
                cuda_toolkit=data["runtime"].get("cuda_toolkit"),
                gpu_name=str(data["runtime"]["gpu_name"]),
                compute_capability=(capability[0], capability[1]),
                gpu_count=int(data["runtime"]["gpu_count"]),
                nccl=data["runtime"].get("nccl"),
                vllm_source_tag=str(data["vllm"]["source_tag"]),
                vllm_source_commit=str(data["vllm"]["source_commit"]),
                wheel_filename=data["wheel"].get("filename"),
                wheel_sha256=data["wheel"].get("sha256"),
                hf_repo_id=data["wheel"].get("hf_repo_id"),
                hf_revision=data["wheel"].get("hf_revision"),
                overlay_lock=data["overlay"].get("lock"),
                overlay_requirements=data["overlay"].get("requirements_reference"),
                attention_backend=str(data["runtime"]["attention_backend"]),
                wheel_status=str(data["wheel"]["status"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProfileError(f"invalid bootstrap profile data: {error}") from error
        if profile.name != data.get("name") or not profile.name:
            raise ProfileError("profile name must be non-empty")
        if not _HEX_40.fullmatch(profile.vllm_source_commit):
            raise ProfileError("vLLM source commit must be a 40-digit commit")
        if profile.python_abi != f"cp{profile.python_major}{profile.python_minor}":
            raise ProfileError("profile Python ABI does not match target Python")
        if data["wheel"].get("python_abi") != profile.python_abi:
            raise ProfileError("native wheel ABI does not match target Python")
        if profile.status == "target_unvalidated":
            if profile.wheel_status != "pending_kaggle_build" or any(
                (
                    profile.wheel_filename,
                    profile.wheel_sha256,
                    profile.hf_repo_id,
                    profile.hf_revision,
                    profile.overlay_lock,
                )
            ):
                raise ProfileError(
                    "unvalidated target must not register a native artifact or overlay"
                )
        elif profile.status == "functionally_validated":
            if not profile.native_ready:
                raise ProfileError(
                    "validated profile requires complete native artifact metadata"
                )
            if not _HEX_40.fullmatch(profile.hf_revision or ""):
                raise ProfileError(
                    "Hugging Face revision must be an immutable 40-digit commit"
                )
            if not _HEX_64.fullmatch(profile.wheel_sha256 or ""):
                raise ProfileError(
                    "wheel SHA256 must be a lowercase 64-digit digest"
                )
            if not profile.wheel_filename or not profile.wheel_filename.endswith(".whl"):
                raise ProfileError("profile wheel filename must end in .whl")
            if f"-{profile.python_abi}-{profile.python_abi}-" not in profile.wheel_filename:
                raise ProfileError(
                    "native wheel filename ABI does not match target Python"
                )
        else:
            raise ProfileError(f"unsupported profile status: {profile.status}")
        return profile


def profile_resource(profile_name: str, filename: str) -> Any:
    """Return a profile resource without extracting or reading credentials."""

    return resources.files("kaggle_vllm").joinpath("profiles", profile_name, filename)


def load_profile(profile_name: str = DEFAULT_PROFILE) -> BootstrapProfile:
    """Load a named profile from lightweight package data."""

    resource = profile_resource(profile_name, "profile.json")
    try:
        data = json.loads(resource.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        raise ProfileError(
            f"unable to load profile {profile_name!r}: {error}"
        ) from error
    profile = BootstrapProfile.from_mapping(data)
    if profile.name != profile_name:
        raise ProfileError(
            f"profile directory {profile_name!r} contains profile {profile.name!r}"
        )
    return profile
