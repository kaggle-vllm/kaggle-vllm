"""Explicit wheel and dependency-overlay staging operations."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from .checksums import verify_sha256
from .exceptions import InstallationError

TORCH_REQUIREMENT = re.compile(
    r"^(torch|torchvision|torchaudio)(?:\[|\s|[<>=!~;@]|$)", re.IGNORECASE
)


def _require_empty_target(target: Path) -> None:
    if target.exists() and not target.is_dir():
        raise InstallationError(
            f"staging target exists and is not a directory: {target}"
        )
    if target.exists() and any(target.iterdir()):
        raise InstallationError(
            f"refusing to overwrite non-empty staging target: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)


def stage_wheel(
    wheel: str | Path,
    target: str | Path,
    *,
    expected_sha256: str | None = None,
    python_executable: str = sys.executable,
) -> Path:
    """Stage a wheel with ``pip --target --no-deps`` and never resolve Torch."""

    wheel_path = Path(wheel).resolve()
    target_path = Path(target).resolve()
    if not wheel_path.is_file() or wheel_path.suffix != ".whl":
        raise InstallationError(
            f"wheel does not exist or lacks a .whl suffix: {wheel_path}"
        )
    if expected_sha256:
        verify_sha256(wheel_path, expected_sha256)
    _require_empty_target(target_path)
    command = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(target_path),
        "--no-deps",
        "--no-cache-dir",
        str(wheel_path),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise InstallationError(f"wheel staging failed: {error}") from error
    return target_path


def _active_requirement_lines(requirements: Path) -> list[str]:
    return [
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def stage_dependency_overlay(
    requirements: str | Path,
    target: str | Path,
    *,
    python_executable: str = sys.executable,
) -> Path:
    """Install an explicit lock into an isolated target while rejecting Torch lines."""

    requirements_path = Path(requirements).resolve()
    target_path = Path(target).resolve()
    if not requirements_path.is_file():
        raise InstallationError(
            f"requirements file does not exist: {requirements_path}"
        )
    forbidden = [
        line
        for line in _active_requirement_lines(requirements_path)
        if TORCH_REQUIREMENT.match(line)
    ]
    if forbidden:
        raise InstallationError(
            "dependency overlay must not contain torch packages: "
            + ", ".join(forbidden)
        )
    _require_empty_target(target_path)
    command = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(target_path),
        "--no-deps",
        "--no-cache-dir",
        "-r",
        str(requirements_path),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise InstallationError(
            f"dependency overlay staging failed: {error}"
        ) from error
    return target_path


def build_runtime_environment(
    staged: str | Path,
    overlay: str | Path | None = None,
    *,
    torch_library: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build, but do not apply, environment variables for staged runtime use."""

    base = dict(os.environ if environ is None else environ)
    python_paths = [str(Path(overlay).resolve())] if overlay is not None else []
    python_paths.append(str(Path(staged).resolve()))
    if base.get("PYTHONPATH"):
        python_paths.append(base["PYTHONPATH"])
    base["PYTHONPATH"] = os.pathsep.join(python_paths)

    executable_paths = [str(Path(staged).resolve() / "bin")]
    if base.get("PATH"):
        executable_paths.append(base["PATH"])
    base["PATH"] = os.pathsep.join(executable_paths)

    library_paths: list[str] = []
    if torch_library is not None:
        library_paths.append(str(Path(torch_library).resolve()))
    library_paths.extend(["/usr/local/nvidia/lib64", "/usr/local/cuda/lib64"])
    if base.get("LD_LIBRARY_PATH"):
        library_paths.append(base["LD_LIBRARY_PATH"])
    base["LD_LIBRARY_PATH"] = os.pathsep.join(library_paths)
    return base
