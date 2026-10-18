"""Validation for an explicitly selected Kaggle global CPython 3.12 runtime."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import platform
import site
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .exceptions import RuntimeValidationError
from .profile_v3 import load_profile


# SHA256 of the unpacked extensions in the pinned P31 native wheel. A wheel
# archive hash alone cannot identify extensions after pip has installed it.
NATIVE_EXTENSIONS = {
    "vllm._C_stable_libtorch": ("vllm/_C_stable_libtorch.abi3.so", "ac72d606840130dd9366366417dc39922b1a1bb2f08c4a3f0a67e25b079fc1cb"),
    "vllm.vllm_flash_attn._vllm_fa2_C": ("vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so", "0299d7b726340fd0cfb936d11d1719608d0b879f0347ef5efa9e0f89c66d1810"),
}


def _global_site_root() -> Path:
    """Kaggle's ordinary root-owned pip destination, never a target overlay."""
    root = Path("/usr/local/lib/python3.12/dist-packages")
    if root not in (Path(p) for p in site.getsitepackages()) or str(root) not in sys.path:
        raise RuntimeValidationError(f"global CPython 3.12 site-packages is not active: {root}")
    if not root.is_dir():
        raise RuntimeValidationError(f"global site-packages directory is missing: {root}")
    return root.resolve()


def _inside(path: str | Path, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_global_runtime() -> dict[str, Any]:
    """Verify the installed L4 stack without staging or downloading packages.

    Native imports and LLM attribute loading remain separate so an optional
    dependency error can be reported at the actual failing boundary.
    """
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise RuntimeValidationError("global L4 mode requires CPython 3.12 / cp312; Python 3.13 is unsupported")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeValidationError("global L4 mode requires Linux x86_64")
    if sys.prefix != sys.base_prefix:
        raise RuntimeValidationError("global L4 mode requires ordinary global site-packages, not a virtual environment")
    from . import l4_bootstrap
    if l4_bootstrap._ACTIVE_L4_TARGET is not None:
        raise RuntimeValidationError("a managed L4 runtime is already active; restart Python before selecting global mode")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "VLLM_NO_USAGE_STATS"):
        os.environ[key] = "1"
    root = _global_site_root()
    profile = load_profile()
    expected = {
        "kaggle-vllm": "0.3.2",
        "vllm": profile.vllm_version,
        "torch": profile.torch_version,
        "triton": profile.data["runtime"]["triton"],
        "torchvision": profile.data["runtime"]["torchvision"],
        "torchaudio": profile.data["runtime"]["torchaudio"],
        "torchcodec": profile.data["runtime"]["torchcodec"],
        "flashinfer-python": profile.data["runtime"]["flashinfer_python"],
        "flashinfer-cubin": profile.data["runtime"]["flashinfer_cubin"],
    }
    installed: dict[str, str] = {}
    for dist in metadata.distributions(path=[str(root)]):
        name = dist.metadata.get("Name", "").lower().replace("_", "-")
        if name in expected:
            if name in installed:
                raise RuntimeValidationError(f"multiple global {name} distributions found in {root}")
            installed[name] = dist.version
    for name, version in expected.items():
        if installed.get(name) != version:
            raise RuntimeValidationError(f"global {name} must be {version} in {root}; found {installed.get(name)!r}")
    for name in ("kaggle_vllm", "vllm", "torch", "triton"):
        spec = importlib.util.find_spec(name)
        origin = getattr(spec, "origin", None)
        if not origin or not _inside(origin, root):
            raise RuntimeValidationError(f"{name} resolves outside global CPython 3.12 site-packages: {origin!r}")
        loaded = sys.modules.get(name)
        if loaded is not None and not _inside(getattr(loaded, "__file__", ""), root):
            raise RuntimeValidationError(f"{name} was already imported outside {root}; restart Python")
    from . import __version__ as sdk_version
    if sdk_version != expected["kaggle-vllm"]:
        raise RuntimeValidationError(f"SDK module version mismatch: {sdk_version}")
    for name in ("vllm", "triton"):
        try:
            module = importlib.import_module(name)
        except Exception as error:
            raise RuntimeValidationError(f"global {name} import failed: {type(error).__name__}: {error}") from error
        if not _inside(getattr(module, "__file__", ""), root):
            raise RuntimeValidationError(f"global {name} imported outside {root}")
        if str(getattr(module, "__version__", "")) != expected[name]:
            raise RuntimeValidationError(f"global {name} module version mismatch: {getattr(module, '__version__', None)}")
    try:
        torch = importlib.import_module("torch")
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeValidationError(f"pinned global torch import failed: {type(error).__name__}: {error}") from error
    if not _inside(getattr(torch, "__file__", ""), root):
        raise RuntimeValidationError(f"global torch imported outside {root}")
    if str(torch.__version__) != profile.torch_version or getattr(torch.version, "cuda", None) != profile.torch_cuda:
        raise RuntimeValidationError(f"global torch/CUDA identity mismatch: {torch.__version__}, CUDA {getattr(torch.version, 'cuda', None)}")
    native: dict[str, str] = {}
    for name, (relative, expected_hash) in NATIVE_EXTENSIONS.items():
        path = root / relative
        if not path.is_file() or not _inside(path, root):
            raise RuntimeValidationError(f"pinned native extension is missing or outside global site-packages: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeValidationError(f"native extension SHA256 mismatch for {path}: {actual_hash}")
        try:
            module = importlib.import_module(name)
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeValidationError(f"native extension {name} import failed: {type(error).__name__}: {error}") from error
        if not _inside(getattr(module, "__file__", ""), root):
            raise RuntimeValidationError(f"native extension {name} imported outside {root}")
        native[name] = actual_hash
    return {"site_packages": str(root), "versions": installed, "torch_cuda": profile.torch_cuda, "native_sha256": native, "native_wheel_sha256": profile.wheel_sha256}
