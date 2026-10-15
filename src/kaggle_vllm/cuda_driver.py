"""SDK-owned CUDA driver linker alias for FlashInfer JIT."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .exceptions import RuntimeValidationError

DEFAULT_DRIVER = Path("/usr/local/nvidia/lib64/libcuda.so.1")
DEFAULT_ALIAS_DIR = Path("/tmp/kaggle-vllm/v0.3.0/l4x4/cuda-driver-link")
SDK_ROOT = DEFAULT_ALIAS_DIR.parent


def ensure_cuda_driver_linker_alias(
    alias_dir: str | Path = DEFAULT_ALIAS_DIR,
    *,
    driver: str | Path = DEFAULT_DRIVER,
    probe: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Create only an SDK-owned symlink; never alter the live driver."""
    root = Path(alias_dir).expanduser()
    canonical = Path(os.path.abspath(root))
    owned = SDK_ROOT in canonical.parents or "kaggle-vllm-v030-runtime" in canonical.parts
    if root.resolve(strict=False) != canonical or not (owned and canonical.name == "cuda-driver-link") or Path("/kaggle/working") in canonical.parents:
        raise RuntimeValidationError("CUDA linker alias must remain under the SDK-owned scratch runtime")
    driver_path = Path(driver)
    real = driver_path.resolve(strict=False)
    if driver_path.name != "libcuda.so.1" or not driver_path.is_file():
        raise RuntimeValidationError(f"live CUDA driver unavailable: {driver_path}")
    alias = canonical / "libcuda.so"
    if not dry_run:
        canonical.mkdir(parents=True, exist_ok=True)
        if alias.is_symlink():
            if alias.resolve() != real:
                raise RuntimeValidationError(f"linker alias points at another driver: {alias}")
        elif alias.exists():
            raise RuntimeValidationError(f"refusing to replace existing alias: {alias}")
        else:
            alias.symlink_to(driver_path)
    library_path = os.pathsep.join(x for x in (str(canonical), os.environ.get("LIBRARY_PATH", "")) if x)
    ld_library_path = os.pathsep.join(x for x in (str(real.parent), os.environ.get("LD_LIBRARY_PATH", "")) if x)
    status = "not_run"
    if probe and not dry_run:
        compiler = shutil.which("g++")
        if compiler:
            with tempfile.TemporaryDirectory(dir=canonical) as tmp:
                source = Path(tmp) / "probe.cc"
                source.write_text('extern "C" int cuInit(unsigned int); extern "C" int probe(){return cuInit(0);}\n', encoding="utf-8")
                result = subprocess.run([compiler, "-shared", "-x", "c++", str(source), "-L", str(canonical), "-lcuda", "-o", str(Path(tmp) / "probe.so")], capture_output=True, text=True)
                status = "PASS" if result.returncode == 0 else "FAIL"
        else:
            status = "g++ unavailable"
    return {"status": "planned" if dry_run else ("FAIL" if status == "FAIL" else "PASS"), "real_driver": str(real), "driver_path": str(driver_path), "alias": str(alias), "LIBRARY_PATH": library_path, "LD_LIBRARY_PATH": ld_library_path, "link_probe": status}
