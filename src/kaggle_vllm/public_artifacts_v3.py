"""Immutable public artifact delivery for the L4 profile.

The revision deliberately remains unset until the additive HF upload is
finished. No request may resolve mutable ``main``.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from .exceptions import BootstrapError
from .profile_v3 import load_profile
from .runtime_v3 import WHEEL_FILENAME, TORCHCODEC_AUDIT_SHA256, lean_runtime_manifest, validate_native_wheel, verify_runtime


def immutable_revision() -> str:
    value = load_profile().data["public_artifacts"].get("immutable_revision")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise BootstrapError("public L4 native wheel and locked runtime dependencies are not published at an immutable Hugging Face revision; attach the exact offline Kaggle Inputs")
    return value


def _download(relative: str, destination: Path, size: int, digest: str) -> None:
    if destination.is_file():
        if destination.stat().st_size == size and hashlib.file_digest(destination.open("rb"), "sha256").hexdigest() == digest:
            return
        raise BootstrapError(f"cached artifact identity mismatch: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    revision = immutable_revision()
    profile = load_profile().data["public_artifacts"]
    remote = profile["namespace"].strip("/") + "/" + relative
    url = f"https://huggingface.co/{quote(profile['hf_repo_id'], safe='/')}/resolve/{revision}/{quote(remote, safe='/')}"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as out:
            temporary = Path(out.name)
            with urlopen(Request(url, headers={"User-Agent": "kaggle-vllm/0.3.1"}), timeout=120) as response:
                while block := response.read(4 * 1024 * 1024):
                    out.write(block)
        if temporary.stat().st_size != size or hashlib.file_digest(temporary.open("rb"), "sha256").hexdigest() != digest:
            raise BootstrapError(f"downloaded artifact identity mismatch: {relative}")
        os.replace(temporary, destination)
    except (OSError, ValueError) as error:
        raise BootstrapError(f"immutable artifact download failed for {relative}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fetch_public_runtime(cache_root: str | Path) -> Path:
    """Cache the exact P33 lean wheel selection; never resolve dependencies."""
    immutable_revision()  # refuse incomplete publication before any file I/O
    root = Path(cache_root).expanduser().resolve() / "kaggle-vllm-v030-l4x4-lean-runtime"
    if root.is_symlink():
        raise BootstrapError("public artifact cache cannot be a symlink")
    lock = lean_runtime_manifest()
    source = Path(__file__).parent / "profiles" / "kaggle-l4x4-cu129" / "L4_INFERENCE_LOCK.json"
    raw = source.read_bytes()
    _download("metadata/L4_INFERENCE_LOCK.json", root / "metadata" / "RUNTIME_MANIFEST.json", len(raw), hashlib.sha256(raw).hexdigest())
    audit = Path(__file__).parent / "profiles" / "kaggle-l4x4-cu129" / "TORCHCODEC_ELF_AUDIT.json"
    audit_target = root / "metadata" / audit.name
    audit_target.write_bytes(audit.read_bytes()) if not audit_target.exists() else None
    if hashlib.file_digest(audit_target.open("rb"), "sha256").hexdigest() != TORCHCODEC_AUDIT_SHA256:
        raise BootstrapError("packaged TorchCodec audit identity mismatch")
    for record in lock["wheels"]:
        _download("wheelhouse/" + record["filename"], root / "wheelhouse" / record["filename"], record["size"], record["sha256"])
    wheel = root / "native" / WHEEL_FILENAME
    identity = load_profile().data["vllm"]
    _download("native/" + WHEEL_FILENAME, wheel, identity["wheel_size"], identity["wheel_sha256"])
    validate_native_wheel(wheel)
    verify_runtime(root)
    return root
