"""Explicit offline staging of the P31 L4/cu129 runtime."""
from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .cuda_driver import ensure_cuda_driver_linker_alias
from .exceptions import BootstrapError
from .profile_v3 import load_profile
from .runtime_v3 import BUNDLE_NAME, SDK_OWNED_ROOT, SplitRuntimeSource, _owned_destination, discover_runtime, extract_runtime, lean_runtime_manifest, load_runtime_manifest, stage_split_runtime, verify_runtime

DEFAULT_TARGET = SDK_OWNED_ROOT
MANIFEST_NAME = "sdk-runtime.json"
MIN_FREE_BYTES = 24 * 1024**3
_ACTIVE_L4_TARGET: Path | None = None


def check_free_space(target: str | Path, *, required: int = MIN_FREE_BYTES) -> dict[str, int]:
    destination = _owned_destination(Path(target))
    ancestor = destination
    while not ancestor.exists():
        ancestor = ancestor.parent
    available = shutil.disk_usage(ancestor).free
    if available < required:
        raise BootstrapError(f"insufficient runtime storage at {destination}: available {available} bytes, required {required} bytes; choose an explicit larger runtime root, never /kaggle/working")
    return {"available_bytes": available, "required_bytes": required}


def reset_runtime(target: str | Path, *, yes: bool = False, dry_run: bool = False) -> dict[str, object]:
    destination = _owned_destination(Path(target))
    manifest = _manifest_path(destination)
    if not destination.exists():
        return {"target": str(destination), "action": "absent", "dry_run": dry_run}
    if not manifest.is_file():
        raise BootstrapError(f"refusing reset without SDK runtime ownership manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("profile") != load_profile().name or payload.get("native_sha256") != load_profile().wheel_sha256:
        raise BootstrapError("refusing reset of runtime with different identity")
    if not dry_run:
        if not yes:
            raise BootstrapError("--reset-runtime requires --yes")
        shutil.rmtree(destination)
    return {"target": str(destination), "action": "remove", "dry_run": dry_run}


def _compatibility(strict: bool) -> list[str]:
    p = load_profile()
    problems = []
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        problems.append("native runtime requires CPython 3.12 / cp312")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        problems.append("native runtime requires Linux x86_64")
    if strict:
        from .environment import detect_kaggle
        if not detect_kaggle(): problems.append("Kaggle markers missing")
        try:
            inventory = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=15, check=True,
            )
            devices = [tuple(part.strip() for part in line.split(",", 1)) for line in inventory.stdout.splitlines() if line.strip()]
            if devices != [("NVIDIA L4", "8.9")] * 4:
                problems.append(f"L4 profile requires exactly four NVIDIA L4 / SM89 GPUs; found {devices}")
        except (OSError, subprocess.SubprocessError) as error:
            problems.append(f"cannot inventory Kaggle GPUs: {error}")
    return problems


def _manifest_path(target: Path) -> Path:
    return target / "manifests" / MANIFEST_NAME


def bootstrap(*, runtime_root: str | Path | None = None, native_source: str | Path | None = None, component_overrides: dict[str, str | Path] | None = None, target: str | Path = DEFAULT_TARGET, strict: bool = False, dry_run: bool = False) -> dict[str, Any]:
    """Verify exact local wheels and install them with --target --no-deps --no-index."""
    selected = discover_runtime(runtime_root, native_source=native_source, component_overrides=component_overrides)
    destination = Path(target).expanduser().resolve()
    problems = _compatibility(strict)
    if not dry_run:
        _owned_destination(Path(target))
        if problems:
            raise BootstrapError("incompatible native runtime: " + "; ".join(problems))
    source_label = str(selected.root if isinstance(selected, SplitRuntimeSource) else selected)
    completed = _manifest_path(destination)
    if not dry_run and completed.is_file():
        old = json.loads(completed.read_text(encoding="utf-8"))
        if old.get("native_sha256") != load_profile().wheel_sha256 or old.get("profile") != load_profile().name:
            raise BootstrapError(f"existing runtime manifest differs: {completed}")
        verified = verify_runtime(Path(old["source"]))
        if not (destination / "pydeps").is_dir():
            raise BootstrapError("completed runtime Python target is unavailable")
        return {"profile": load_profile().name, "source": source_label, "target": str(destination), "strict": strict, "compatible": not problems, "findings": problems, "wheel_count": len(verified["wheels"]), "dry_run": False, "already_complete": True}
    if not dry_run:
        check_free_space(destination)
        destination.mkdir(parents=True, exist_ok=True)
    if isinstance(selected, Path) and selected.is_file():
        source = destination / "source" / BUNDLE_NAME
        if dry_run:
            manifest = lean_runtime_manifest()
        else:
            source = source if source.is_dir() else extract_runtime(selected, destination / "source")
            manifest = verify_runtime(source)
    elif isinstance(selected, SplitRuntimeSource):
        source = destination / "source" / BUNDLE_NAME
        if dry_run:
            manifest = lean_runtime_manifest()
        elif source.is_dir():
            manifest = verify_runtime(source)
        else:
            manifest = verify_runtime(stage_split_runtime(selected, source))
    else:
        source = selected
        manifest = load_runtime_manifest(source) if dry_run else verify_runtime(source)
    result = {"profile": load_profile().name, "source": source_label, "target": str(destination), "strict": strict, "compatible": not problems, "findings": problems, "wheel_count": len(manifest["wheels"]), "dry_run": dry_run}
    if dry_run:
        return result
    pydeps = destination / "pydeps"
    if pydeps.exists() and any(pydeps.iterdir()):
        raise BootstrapError(f"refusing nonempty staging target: {pydeps}")
    wheels = [source / "wheelhouse" / r["filename"] for r in manifest["wheels"]]
    if {wheel.name for wheel in wheels} != {record["filename"] for record in lean_runtime_manifest()["wheels"]}:
        raise BootstrapError("runtime source does not contain the exact P33 lean inference selection")
    wheels.append(source / "native" / load_profile().wheel_filename)
    # One pip invocation avoids overwriting shared package directories across calls.
    temporary_pydeps = Path(tempfile.mkdtemp(prefix=".pydeps-", dir=destination))
    command = [sys.executable, "-m", "pip", "install", "--target", str(temporary_pydeps), "--no-deps", "--no-index", "--no-cache-dir", "--no-compile", "--disable-pip-version-check", "--quiet", *map(str, wheels)]
    log_dir = destination / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "bootstrap-pip.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        driver = ensure_cuda_driver_linker_alias(destination / "cuda-driver-link", probe=True)
        os.replace(temporary_pydeps, pydeps)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise BootstrapError(f"offline wheel staging failed: {error}; see {log_path}") from error
    finally:
        if temporary_pydeps.exists():
            shutil.rmtree(temporary_pydeps)
    env = runtime_environment(destination, driver)
    completed.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 3, "profile": load_profile().name, "source": str(source), "native_sha256": load_profile().wheel_sha256, "wheel_count": len(manifest["wheels"]), "pydeps": str(pydeps), "driver": driver, "environment": env}
    temporary = completed.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, completed)
    result.update({"completed": True, "manifest": str(completed), "driver": driver})
    return result


def _paths(*values: str) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        for entry in value.split(os.pathsep):
            if entry and entry not in seen:
                seen.add(entry)
                result.append(entry)
    return os.pathsep.join(result)


def runtime_environment(target: str | Path = DEFAULT_TARGET, driver: dict[str, object] | None = None) -> dict[str, str]:
    dest = Path(target).resolve()
    alias = dest / "cuda-driver-link"
    return {
        "PYTHONPATH": _paths(str(dest / "pydeps"), os.environ.get("PYTHONPATH", "")),
        "LIBRARY_PATH": _paths(str(alias), os.environ.get("LIBRARY_PATH", "")),
        "LD_LIBRARY_PATH": _paths("/usr/local/nvidia/lib64", str(dest / "pydeps" / "torch" / "lib"), os.environ.get("LD_LIBRARY_PATH", "")),
    }


def shell_exports(target: str | Path = DEFAULT_TARGET) -> tuple[str, ...]:
    path = _manifest_path(Path(target).resolve())
    if not path.is_file():
        raise BootstrapError(f"bootstrap manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 3 or payload.get("native_sha256") != load_profile().wheel_sha256:
        raise BootstrapError("bootstrap manifest identity mismatch")
    return tuple(f"export {name}={shlex.quote(value)}" for name, value in runtime_environment(target).items())


def activate_runtime(target: str | Path = DEFAULT_TARGET) -> bool:
    global _ACTIVE_L4_TARGET
    destination = Path(target).resolve()
    if _ACTIVE_L4_TARGET is not None and _ACTIVE_L4_TARGET != destination:
        raise BootstrapError(f"another L4 runtime is active at {_ACTIVE_L4_TARGET}; restart the Python process")
    path = _manifest_path(destination)
    if not path.is_file():
        return False
    problems = _compatibility(False)
    if problems:
        raise BootstrapError("incompatible native runtime: " + "; ".join(problems))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 3 or payload.get("native_sha256") != load_profile().wheel_sha256:
        raise BootstrapError("bootstrap manifest identity mismatch")
    pydeps = Path(payload["pydeps"])
    if not pydeps.is_dir():
        raise BootstrapError("staged Python dependencies missing")
    for name in ("torch", "vllm", "triton", "torchcodec", "flashinfer"):
        module = sys.modules.get(name)
        if module is not None:
            location = getattr(module, "__file__", None)
            if location is None or pydeps.resolve() not in Path(location).resolve().parents:
                raise BootstrapError(f"{name} was imported outside the SDK runtime; restart the kernel and activate first")
    value = str(pydeps)
    if value in sys.path: sys.path.remove(value)
    sys.path.insert(0, value)
    os.environ.update(runtime_environment(target))
    _ACTIVE_L4_TARGET = destination
    return True
