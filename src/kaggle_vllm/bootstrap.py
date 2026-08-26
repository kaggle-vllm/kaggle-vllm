"""Explicit bootstrap of the validated Kaggle native vLLM runtime."""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
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
class ResetTarget:
    """One filesystem target in a validated runtime-reset plan."""

    label: str
    path: Path
    exists: bool
    owned: bool
    action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": str(self.path),
            "exists": self.exists,
            "owned": self.owned,
            "action": self.action,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ResetPlan:
    """Validated, side-effect-free plan for removing SDK-owned runtime state."""

    paths: BootstrapPaths
    targets: tuple[ResetTarget, ...]

    @property
    def safe(self) -> bool:
        return all(
            target.action in {"remove", "preserve", "absent"} for target in self.targets
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "targets": [target.to_dict() for target in self.targets],
        }


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
    reset_plan: ResetPlan | None = None
    reset_completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = self.plan.to_dict()
        result.update(
            {
                "manifest": str(self.manifest) if self.manifest else None,
                "completed": self.completed,
                "already_complete": self.already_complete,
                "reset": (
                    {
                        **self.reset_plan.to_dict(),
                        "completed": self.reset_completed,
                    }
                    if self.reset_plan
                    else None
                ),
            }
        )
        return result


def _finding(
    check: str, matches: bool, expected: str, actual: str, strict: bool
) -> CompatibilityFinding:
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
        _finding(
            "operating system",
            current_system == profile.system,
            profile.system,
            current_system,
            True,
        ),
        _finding(
            "machine",
            current_machine == profile.machine,
            profile.machine,
            current_machine,
            True,
        ),
        _finding(
            "Kaggle runtime",
            environment.is_kaggle,
            "Kaggle",
            str(environment.is_kaggle),
            strict,
        ),
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
            bool(environment.gpus)
            and all(
                profile.gpu_name.casefold() in gpu.name.casefold()
                or gpu.name.casefold() in profile.gpu_name.casefold()
                for gpu in environment.gpus
            ),
            profile.gpu_name,
            ", ".join(environment.gpu_names) or "none",
            strict,
        ),
        _finding(
            "GPU compute capability",
            bool(environment.gpus)
            and all(
                gpu.capability == profile.compute_capability for gpu in environment.gpus
            ),
            f"SM{profile.compute_capability[0]}{profile.compute_capability[1]}",
            ", ".join(f"SM{major}{minor}" for major, minor in environment.gpu_caps)
            or "none",
            strict,
        ),
        _finding(
            "NCCL",
            environment.nccl == profile.nccl,
            profile.nccl,
            str(environment.nccl),
            strict,
        ),
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


def _repository_roots() -> tuple[Path, ...]:
    roots: set[Path] = set()
    for seed in (Path.cwd().resolve(), Path(__file__).resolve()):
        for candidate in (seed, *seed.parents):
            if (candidate / ".git").exists():
                roots.add(candidate)
                break
    return tuple(sorted(roots, key=str))


def validate_reset_path(path: str | Path, *, label: str) -> Path:
    """Resolve a reset target and reject dangerous or symlinked selections."""

    raw = os.fspath(path)
    if not raw.strip():
        raise BootstrapError(f"refusing empty {label} reset path")
    expanded = Path(raw).expanduser()
    lexical = Path(os.path.abspath(expanded))
    resolved = expanded.resolve(strict=False)
    if lexical != resolved:
        raise BootstrapError(
            f"refusing {label} reset path that traverses a symlink: {lexical} -> {resolved}"
        )

    dangerous = {
        Path("/"),
        Path.home().resolve(),
        Path("/root"),
        Path("/home"),
        Path("/usr"),
        Path("/usr/local"),
        Path("/opt"),
        Path("/etc"),
        Path("/var"),
        Path("/tmp"),
        Path("/kaggle"),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
    }
    if resolved in dangerous:
        raise BootstrapError(f"refusing dangerous {label} reset path: {resolved}")

    current = Path.cwd().resolve()
    if resolved == current or resolved in current.parents:
        raise BootstrapError(
            f"refusing {label} reset path that is the current directory or its parent: {resolved}"
        )
    for repository in _repository_roots():
        if resolved == repository or repository in resolved.parents:
            raise BootstrapError(
                f"refusing {label} reset path inside repository root {repository}: {resolved}"
            )
    return resolved


def _expected_manifest_paths(paths: BootstrapPaths) -> dict[str, str]:
    return {
        "staged": str(paths.staged),
        "overlay": str(paths.overlay),
        "cache": str(paths.cache),
        "manifest": str(paths.manifest),
    }


def build_reset_plan(
    *,
    staged: str | Path = DEFAULT_STAGED,
    overlay: str | Path = DEFAULT_OVERLAY,
    cache: str | Path = DEFAULT_CACHE,
    manifest: str | Path = DEFAULT_MANIFEST,
    profile: BootstrapProfile | None = None,
) -> ResetPlan:
    """Validate ownership and return a reset plan without changing the filesystem."""

    selected_profile = profile or load_profile(DEFAULT_PROFILE)
    paths = BootstrapPaths(
        staged=validate_reset_path(staged, label="staged"),
        overlay=validate_reset_path(overlay, label="overlay"),
        cache=validate_reset_path(cache, label="cache"),
        manifest=validate_reset_path(manifest, label="manifest"),
    )
    selected = {
        "staged": paths.staged,
        "overlay": paths.overlay,
        "manifest": paths.manifest,
    }
    all_paths = {**selected, "cache": paths.cache}
    if len(set(all_paths.values())) != len(all_paths):
        raise BootstrapError("reset paths must be distinct")
    for label, candidate in selected.items():
        for other_label, other in all_paths.items():
            if label != other_label and (
                candidate in other.parents or other in candidate.parents
            ):
                raise BootstrapError(
                    f"refusing {label} reset path because it overlaps the selected "
                    f"{other_label} path: {candidate}"
                )

    manifest_matches = False
    if paths.manifest.exists():
        if not paths.manifest.is_file():
            raise BootstrapError(f"runtime manifest is not a file: {paths.manifest}")
        data = _load_manifest(paths.manifest)
        identity_matches = (
            data.get("schema_version") == 1
            and data.get("profile") == selected_profile.name
            and data.get("wheel", {}).get("sha256") == selected_profile.wheel_sha256
            and data.get("wheel", {}).get("hf_revision") == selected_profile.hf_revision
        )
        if not identity_matches or data.get("paths") != _expected_manifest_paths(paths):
            raise BootstrapError(
                "existing manifest identity or paths do not match the selected reset: "
                f"{paths.manifest}"
            )
        manifest_matches = True

    known_defaults = {
        "staged": DEFAULT_STAGED.resolve(),
        "overlay": DEFAULT_OVERLAY.resolve(),
        "manifest": DEFAULT_MANIFEST.resolve(),
    }
    targets: list[ResetTarget] = []
    for label in ("staged", "overlay"):
        candidate = selected[label]
        exists = candidate.exists()
        if exists and not candidate.is_dir():
            raise BootstrapError(
                f"selected {label} runtime path is not a directory: {candidate}"
            )
        owned = candidate == known_defaults[label] or manifest_matches
        if exists and not owned:
            raise BootstrapError(
                f"cannot prove ownership of custom {label} runtime path without "
                f"a matching manifest: {candidate}"
            )
        reason = (
            "matching runtime manifest"
            if manifest_matches
            else (
                "known SDK default path"
                if candidate == known_defaults[label]
                else "path is absent; nothing to remove"
            )
        )
        targets.append(
            ResetTarget(
                label,
                candidate,
                exists,
                owned,
                "remove" if exists else "absent",
                reason,
            )
        )

    manifest_exists = paths.manifest.exists()
    manifest_owned = paths.manifest == known_defaults["manifest"] or manifest_matches
    if manifest_exists and not manifest_owned:
        raise BootstrapError(
            f"cannot prove ownership of runtime manifest: {paths.manifest}"
        )
    manifest_reason = (
        "matching runtime manifest"
        if manifest_matches
        else (
            "known SDK default path"
            if paths.manifest == known_defaults["manifest"]
            else "path is absent; nothing to remove"
        )
    )
    targets.append(
        ResetTarget(
            "manifest",
            paths.manifest,
            manifest_exists,
            manifest_owned,
            "remove" if manifest_exists else "absent",
            manifest_reason,
        )
    )
    targets.append(
        ResetTarget(
            "cache",
            paths.cache,
            paths.cache.exists(),
            False,
            "preserve",
            "download cache is preserved by default",
        )
    )
    return ResetPlan(paths, tuple(targets))


def execute_reset(plan: ResetPlan) -> tuple[Path, ...]:
    """Remove only validated reset targets and return the paths removed."""

    removed: list[Path] = []
    for target in plan.targets:
        if target.action != "remove":
            continue
        try:
            if validate_reset_path(target.path, label=target.label) != target.path:
                raise BootstrapError(
                    f"reset target changed after planning: {target.path}"
                )
            if target.path.is_symlink():
                raise BootstrapError(
                    f"refusing reset target that became a symlink: {target.path}"
                )
            if target.label in {"staged", "overlay"}:
                shutil.rmtree(target.path)
            else:
                target.path.unlink()
            removed.append(target.path)
        except (OSError, BootstrapError) as error:
            prior = ", ".join(str(path) for path in removed) or "none"
            raise BootstrapError(
                f"runtime reset failed at {target.label} path {target.path}: {error}; "
                f"already removed: {prior}; cache preserved: {plan.paths.cache}"
            ) from error
    return tuple(removed)


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
        raise BootstrapError(
            f"refusing to overwrite non-empty {label} destination: {path}"
        )


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
    identity_matches = (
        data.get("profile") == plan.profile.name
        and data.get("wheel", {}).get("sha256") == plan.profile.wheel_sha256
        and data.get("wheel", {}).get("hf_revision") == plan.profile.hf_revision
        and data.get("paths") == _expected_manifest_paths(plan.paths)
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
    reset_runtime: bool = False,
    yes: bool = False,
    environment: Environment | None = None,
    python_executable: str = sys.executable,
    implementation: str | None = None,
    python_version: tuple[int, int] | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> BootstrapResult:
    """Explicitly reset if requested, then download and stage the native runtime."""

    if yes and not reset_runtime:
        raise BootstrapError("--yes is only valid together with --reset-runtime")

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
    reset_plan = (
        build_reset_plan(
            staged=staged,
            overlay=overlay,
            cache=cache,
            manifest=manifest,
            profile=plan.profile,
        )
        if reset_runtime
        else None
    )
    if dry_run:
        return BootstrapResult(plan, None, False, False, reset_plan, False)
    errors = [finding.message for finding in plan.findings if finding.status == "error"]
    if errors:
        raise BootstrapError("incompatible native runtime:\n- " + "\n- ".join(errors))
    reset_completed = False
    if reset_plan:
        if not yes:
            raise BootstrapError(
                "runtime reset requires explicit --yes confirmation; "
                "inspect it first with --reset-runtime --dry-run"
            )
        execute_reset(reset_plan)
        reset_completed = True
    if _matching_manifest(plan):
        return BootstrapResult(
            plan,
            plan.paths.manifest,
            True,
            True,
            reset_plan,
            reset_completed,
        )
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
    return BootstrapResult(
        plan,
        manifest_path,
        True,
        False,
        reset_plan,
        reset_completed,
    )


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
        raise BootstrapError(
            f"runtime manifest lacks activation data: {manifest}"
        ) from error


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
    return tuple(
        f"export {key}={shlex.quote(value)}" for key, value in environment.items()
    )
