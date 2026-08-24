"""Explicit bootstrap of the validated Kaggle native vLLM runtime."""

from __future__ import annotations

import json
import os
import platform
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

from .download import download_wheel, immutable_resolve_url
from .environment import Environment, collect
from .exceptions import BootstrapError
from .installation import (
    build_runtime_environment,
    stage_dependency_overlay,
    stage_wheel,
)
from .profiles import DEFAULT_PROFILE, BootstrapProfile, load_profile, profile_resource
from .runtime import all_sm75, all_tesla_t4

DEFAULT_STAGED = Path("/kaggle/working/vllm-staged")
DEFAULT_OVERLAY = Path("/kaggle/working/vllm-runtime-overlay")
DEFAULT_CACHE = Path("/kaggle/working/kaggle-vllm-cache")
DEFAULT_MANIFEST = Path("/kaggle/working/kaggle-vllm-runtime.json")
MANIFEST_ENVIRONMENT_VARIABLE = "KAGGLE_VLLM_MANIFEST"


@dataclass(frozen=True)
class CompatibilityFinding:
    """One host/profile comparison result."""

    check: str
    status: str
    message: str


@dataclass(frozen=True)
class BootstrapPaths:
    """Resolved bootstrap destinations."""

    staged: Path
    overlay: Path
    cache: Path
    manifest: Path


@dataclass(frozen=True)
class BootstrapPlan:
    """Side-effect-free plan for a native runtime bootstrap."""

    profile: BootstrapProfile
    paths: BootstrapPaths
    environment: Environment
    findings: tuple[CompatibilityFinding, ...]
    strict: bool
    commands: tuple[tuple[str, ...], ...]

    @property
    def compatible(self) -> bool:
        return not any(finding.status == "error" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.name,
            "strict": self.strict,
            "compatible": self.compatible,
            "findings": [asdict(finding) for finding in self.findings],
            "artifact": {
                "hf_repo_id": self.profile.hf_repo_id,
                "hf_revision": self.profile.hf_revision,
                "filename": self.profile.wheel_filename,
                "sha256": self.profile.wheel_sha256,
                "fallback_url": immutable_resolve_url(self.profile),
            },
            "paths": {key: str(value) for key, value in asdict(self.paths).items()},
            "commands": [list(command) for command in self.commands],
            "environment": asdict(self.environment),
        }


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a dry run, new bootstrap, or idempotent reuse."""

    plan: BootstrapPlan
    manifest: Path | None
    completed: bool
    already_complete: bool

    def to_dict(self) -> dict[str, Any]:
        result = self.plan.to_dict()
        result.update(
            {
                "manifest": str(self.manifest) if self.manifest else None,
                "completed": self.completed,
                "already_complete": self.already_complete,
            }
        )
        return result


def _finding(check: str, matches: bool, expected: str, actual: str, strict: bool) -> CompatibilityFinding:
    if matches:
        return CompatibilityFinding(check, "pass", f"{check}: {actual}")
    status = "error" if strict else "warning"
    return CompatibilityFinding(
        check,
        status,
        f"{check}: expected {expected}; found {actual}",
    )


def inspect_compatibility(
    profile: BootstrapProfile,
    environment: Environment,
    *,
    strict: bool = False,
    implementation: str | None = None,
    python_version: tuple[int, int] | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> tuple[CompatibilityFinding, ...]:
    """Compare a host to a profile without downloading or installing anything."""

    current_implementation = implementation or platform.python_implementation()
    current_python = python_version or (sys.version_info.major, sys.version_info.minor)
    current_system = system or platform.system()
    current_machine = machine or platform.machine()
    findings = [
        _finding(
            "Python implementation",
            current_implementation == profile.python_implementation,
            profile.python_implementation,
            current_implementation,
            True,
        ),
        _finding(
            "Python ABI",
            current_python == (profile.python_major, profile.python_minor),
            profile.python_abi,
            f"cp{current_python[0]}{current_python[1]}",
            True,
        ),
        _finding("operating system", current_system == profile.system, profile.system, current_system, True),
        _finding("machine", current_machine == profile.machine, profile.machine, current_machine, True),
        _finding("Kaggle runtime", environment.is_kaggle, "Kaggle", str(environment.is_kaggle), strict),
        _finding(
            "PyTorch",
            environment.torch == profile.torch_version,
            profile.torch_version,
            str(environment.torch),
            strict,
        ),
        _finding(
            "PyTorch CUDA",
            environment.torch_cuda == profile.torch_cuda,
            profile.torch_cuda,
            str(environment.torch_cuda),
            strict,
        ),
        _finding(
            "visible GPU count",
            environment.gpu_count == profile.gpu_count,
            str(profile.gpu_count),
            str(environment.gpu_count),
            strict,
        ),
        _finding(
            "GPU model",
            all_tesla_t4(environment.gpus),
            profile.gpu_name,
            ", ".join(environment.gpu_names) or "none",
            strict,
        ),
        _finding(
            "GPU compute capability",
            all_sm75(environment.gpus),
            "SM75",
            ", ".join(f"SM{major}{minor}" for major, minor in environment.gpu_caps) or "none",
            strict,
        ),
        _finding("NCCL", environment.nccl == profile.nccl, profile.nccl, str(environment.nccl), strict),
    ]
    return tuple(findings)


def _resolve_paths(
    staged: str | Path,
    overlay: str | Path,
    cache: str | Path,
    manifest: str | Path,
) -> BootstrapPaths:
    return BootstrapPaths(
        staged=Path(staged).expanduser().resolve(),
        overlay=Path(overlay).expanduser().resolve(),
        cache=Path(cache).expanduser().resolve(),
        manifest=Path(manifest).expanduser().resolve(),
    )


def build_bootstrap_plan(
    *,
    profile_name: str = DEFAULT_PROFILE,
    staged: str | Path = DEFAULT_STAGED,
    overlay: str | Path = DEFAULT_OVERLAY,
    cache: str | Path = DEFAULT_CACHE,
    manifest: str | Path = DEFAULT_MANIFEST,
    strict: bool = False,
    environment: Environment | None = None,
    python_executable: str = sys.executable,
    implementation: str | None = None,
    python_version: tuple[int, int] | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> BootstrapPlan:
    """Build a printable plan without changing the filesystem or network."""

    profile = load_profile(profile_name)
    paths = _resolve_paths(staged, overlay, cache, manifest)
    snapshot = environment or collect()
    findings = inspect_compatibility(
        profile,
        snapshot,
        strict=strict,
        implementation=implementation,
        python_version=python_version,
        system=system,
        machine=machine,
    )
    cached_wheel = paths.cache / profile.wheel_filename
    lock_path = str(profile_resource(profile.name, profile.overlay_lock))
    commands = (
        (
            python_executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(paths.staged),
            "--no-deps",
            "--no-cache-dir",
            str(cached_wheel),
        ),
        (
            python_executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(paths.overlay),
            "--no-deps",
            "--no-cache-dir",
            "-r",
            lock_path,
        ),
    )
    return BootstrapPlan(profile, paths, snapshot, findings, strict, commands)


def _check_destination(path: Path, label: str) -> None:
    if path.exists() and not path.is_dir():
        raise BootstrapError(f"{label} destination is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise BootstrapError(f"refusing to overwrite non-empty {label} destination: {path}")


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise BootstrapError(f"invalid runtime manifest {path}: {error}") from error


def _matching_manifest(plan: BootstrapPlan) -> bool:
    path = plan.paths.manifest
    if not path.is_file():
        return False
    data = _load_manifest(path)
    expected_paths = {
        "staged": str(plan.paths.staged),
        "overlay": str(plan.paths.overlay),
        "cache": str(plan.paths.cache),
        "manifest": str(plan.paths.manifest),
    }
    identity_matches = (
        data.get("profile") == plan.profile.name
        and data.get("wheel", {}).get("sha256") == plan.profile.wheel_sha256
        and data.get("wheel", {}).get("hf_revision") == plan.profile.hf_revision
        and data.get("paths") == expected_paths
    )
    directories_ready = all(
        path.is_dir() and any(path.iterdir())
        for path in (plan.paths.staged, plan.paths.overlay)
    )
    if identity_matches and directories_ready:
        return True
    raise BootstrapError(
        f"existing manifest/runtime does not match requested profile and paths: {path}"
    )


def _torch_library(environment: Environment) -> Path | None:
    if not environment.torch_path:
        return None
    candidate = Path(environment.torch_path).resolve().parent / "lib"
    return candidate if candidate.is_dir() else None


def _write_manifest(plan: BootstrapPlan, wheel: Path) -> Path:
    runtime_environment = build_runtime_environment(
        plan.paths.staged,
        plan.paths.overlay,
        torch_library=_torch_library(plan.environment),
        environ={"PATH": os.environ.get("PATH", os.defpath)},
    )
    data = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "profile": plan.profile.name,
        "strict": plan.strict,
        "wheel": {
            "filename": plan.profile.wheel_filename,
            "sha256": plan.profile.wheel_sha256,
            "hf_repo_id": plan.profile.hf_repo_id,
            "hf_revision": plan.profile.hf_revision,
            "resolved_path": str(wheel),
        },
        "paths": {
            "staged": str(plan.paths.staged),
            "overlay": str(plan.paths.overlay),
            "cache": str(plan.paths.cache),
            "manifest": str(plan.paths.manifest),
        },
        "runtime_environment": {
            "PYTHONPATH": runtime_environment["PYTHONPATH"],
            "LD_LIBRARY_PATH": runtime_environment["LD_LIBRARY_PATH"],
            "PATH": runtime_environment["PATH"],
        },
        "environment": asdict(plan.environment),
        "findings": [asdict(finding) for finding in plan.findings],
    }
    plan.paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan.paths.manifest.with_suffix(plan.paths.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, plan.paths.manifest)
    return plan.paths.manifest


def bootstrap(
    *,
    profile_name: str = DEFAULT_PROFILE,
    staged: str | Path = DEFAULT_STAGED,
    overlay: str | Path = DEFAULT_OVERLAY,
    cache: str | Path = DEFAULT_CACHE,
    manifest: str | Path = DEFAULT_MANIFEST,
    strict: bool = False,
    dry_run: bool = False,
    environment: Environment | None = None,
    python_executable: str = sys.executable,
    implementation: str | None = None,
    python_version: tuple[int, int] | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> BootstrapResult:
    """Explicitly download, verify, and stage the profiled native runtime."""

    plan = build_bootstrap_plan(
        profile_name=profile_name,
        staged=staged,
        overlay=overlay,
        cache=cache,
        manifest=manifest,
        strict=strict,
        environment=environment,
        python_executable=python_executable,
        implementation=implementation,
        python_version=python_version,
        system=system,
        machine=machine,
    )
    if dry_run:
        return BootstrapResult(plan, None, False, False)
    errors = [finding.message for finding in plan.findings if finding.status == "error"]
    if errors:
        raise BootstrapError("incompatible native runtime:\n- " + "\n- ".join(errors))
    if _matching_manifest(plan):
        return BootstrapResult(plan, plan.paths.manifest, True, True)
    _check_destination(plan.paths.staged, "staged wheel")
    _check_destination(plan.paths.overlay, "dependency overlay")

    wheel = download_wheel(plan.profile, plan.paths.cache)
    stage_wheel(
        wheel,
        plan.paths.staged,
        expected_sha256=plan.profile.wheel_sha256,
        python_executable=python_executable,
    )
    lock = profile_resource(plan.profile.name, plan.profile.overlay_lock)
    with resources.as_file(lock) as lock_path:
        stage_dependency_overlay(
            lock_path,
            plan.paths.overlay,
            python_executable=python_executable,
        )
    manifest_path = _write_manifest(plan, wheel)
    return BootstrapResult(plan, manifest_path, True, False)


def runtime_manifest_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit, environment-selected, or default manifest path."""

    selected = path or os.environ.get(MANIFEST_ENVIRONMENT_VARIABLE) or DEFAULT_MANIFEST
    return Path(selected).expanduser().resolve()


def runtime_environment_from_manifest(path: str | Path | None = None) -> dict[str, str]:
    """Read only the three non-secret activation variables from a manifest."""

    manifest = runtime_manifest_path(path)
    if not manifest.is_file():
        raise BootstrapError(f"runtime manifest does not exist: {manifest}")
    data = _load_manifest(manifest)
    runtime_environment = data.get("runtime_environment", {})
    try:
        return {
            "PYTHONPATH": str(runtime_environment["PYTHONPATH"]),
            "LD_LIBRARY_PATH": str(runtime_environment["LD_LIBRARY_PATH"]),
            "PATH": str(runtime_environment["PATH"]),
        }
    except KeyError as error:
        raise BootstrapError(f"runtime manifest lacks activation data: {manifest}") from error


def activate_runtime(path: str | Path | None = None) -> bool:
    """Activate a completed bootstrap in this process; never download or install."""

    manifest = runtime_manifest_path(path)
    if not manifest.is_file():
        return False
    data = _load_manifest(manifest)
    paths = data.get("paths", {})
    overlay = Path(str(paths.get("overlay", "")))
    staged = Path(str(paths.get("staged", "")))
    if not overlay.is_dir() or not staged.is_dir():
        raise BootstrapError(f"runtime paths from manifest are unavailable: {manifest}")
    for entry in reversed((str(overlay), str(staged))):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    activation = runtime_environment_from_manifest(manifest)
    os.environ.update(activation)
    return True


def shell_exports(path: str | Path | None = None) -> tuple[str, ...]:
    """Return safely quoted shell exports without modifying startup files."""

    environment = runtime_environment_from_manifest(path)
    return tuple(f"export {key}={shlex.quote(value)}" for key, value in environment.items())
