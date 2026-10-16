"""Checksum-pinned public L4 artifacts. Publication is deliberately fail closed."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .exceptions import BootstrapError
from .profile_v3 import load_profile
from .runtime_v3 import (
    TORCHCODEC_AUDIT_SHA256, WHEEL_FILENAME, lean_runtime_manifest,
    validate_native_wheel, verify_runtime,
)

LOCK_SHA256 = "f162e6c212371a34ed49f11885e80222fb78359a2d9a92f02b3cd29db2ce5329"
CHUNK = 4 * 1024 * 1024
ATTEMPTS = 3


def immutable_revision() -> str:
    value = load_profile().data["public_artifacts"].get("immutable_revision")
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise BootstrapError(
            "public L4 wheelhouse is not complete at a verified immutable Hugging Face revision; "
            "use the documented offline runtime until publication"
        )
    return value


def _safe_root(root: Path) -> Path:
    root = root.expanduser().absolute()
    if root.resolve(strict=False) != root:
        raise BootstrapError(f"public artifact cache cannot traverse symlinks: {root}")
    return root


def _safe_name(name: str) -> str:
    if not name or name in (".", "..") or Path(name).name != name or "\\" in name or "/" in name:
        raise BootstrapError(f"unsafe artifact filename: {name!r}")
    return name


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def _cache_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    fd = os.open(root / ".download.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        for attempt in range(120):
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if attempt == 119:
                    raise BootstrapError("public artifact cache is busy for more than 60 seconds")
                time.sleep(0.5)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _artifact_url(relative: str) -> str:
    revision = immutable_revision()
    profile = load_profile().data["public_artifacts"]
    remote = profile["namespace"].strip("/") + "/" + relative
    return f"https://huggingface.co/{quote(profile['hf_repo_id'], safe='/')}/resolve/{revision}/{quote(remote, safe='/')}"


def _download(relative: str, destination: Path, size: int, digest: str) -> None:
    """Resume one allowlisted artifact, verify it, then atomically activate it."""
    if size < 1 or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise BootstrapError(f"invalid pinned artifact identity: {relative}")
    if destination.is_symlink() or destination.parent.is_symlink():
        raise BootstrapError(f"artifact path is a symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.stat().st_size == size and _sha256(destination) == digest:
            return
        raise BootstrapError(f"cached artifact identity mismatch: {destination}")
    partial = destination.with_name(destination.name + ".part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise BootstrapError(f"unsafe partial download path: {partial}")
    if partial.exists() and partial.stat().st_size > size:
        partial.unlink()
    url = _artifact_url(relative)
    last_error: Exception | None = None
    for attempt in range(ATTEMPTS):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            if _sha256(partial) == digest:
                os.replace(partial, destination)
                return
            partial.unlink()
            offset = 0
        headers = {"User-Agent": "kaggle-vllm/0.3.2"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as response:
                status = response.status
                content_range = response.headers.get("Content-Range", "")
                if offset and (status != 206 or not content_range.startswith(f"bytes {offset}-")):
                    partial.unlink(missing_ok=True)
                    offset = 0
                elif not offset and status != 200:
                    raise BootstrapError(f"unexpected HTTP {status} for {relative}")
                with partial.open("ab" if offset else "wb") as out:
                    while block := response.read(CHUNK):
                        out.write(block)
                        if out.tell() > size:
                            raise BootstrapError(f"download exceeded pinned size: {relative}")
                    out.flush()
                    os.fsync(out.fileno())
            if partial.stat().st_size == size and _sha256(partial) == digest:
                os.replace(partial, destination)
                return
            if partial.stat().st_size == size:
                raise BootstrapError(f"downloaded artifact SHA256 mismatch: {relative}")
            last_error = BootstrapError(f"incomplete download: {relative}")
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            if isinstance(error, HTTPError) and error.code in (400, 401, 403, 404):
                break
        if attempt + 1 < ATTEMPTS:
            time.sleep(min(2**attempt, 4))
    if isinstance(last_error, BootstrapError) and "SHA256 mismatch" in str(last_error):
        partial.unlink(missing_ok=True)
    raise BootstrapError(f"immutable artifact download failed for {relative}: {last_error}")


def _remote_inventory() -> dict[str, dict]:
    """Require the complete lock in one immutable Hub tree before fetching bytes."""
    profile = load_profile().data["public_artifacts"]
    revision = immutable_revision()
    url = (f"https://huggingface.co/api/models/{quote(profile['hf_repo_id'], safe='/')}"
           f"/tree/{revision}?recursive=true")
    try:
        with urlopen(Request(url, headers={"User-Agent": "kaggle-vllm/0.3.2"}), timeout=30) as response:
            rows = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
        raise BootstrapError(f"Hugging Face inventory request failed: {error}") from error
    if not isinstance(rows, list):
        raise BootstrapError("Hugging Face inventory response is invalid")
    return {item["path"]: item for item in rows if isinstance(item, dict) and item.get("type") == "file" and isinstance(item.get("path"), str)}


def _remote_small_sha256(relative: str, size: int) -> str:
    """Verify a Git-stored small wheel by bytes when no LFS digest is exposed."""
    digest = hashlib.sha256()
    seen = 0
    try:
        with urlopen(Request(_artifact_url(relative), headers={"User-Agent": "kaggle-vllm/0.3.2"}), timeout=30) as response:
            while block := response.read(CHUNK):
                seen += len(block)
                if seen > size:
                    raise BootstrapError(f"remote artifact exceeds pinned size: {relative}")
                digest.update(block)
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise BootstrapError(f"cannot verify remote artifact {relative}: {error}") from error
    if seen != size:
        raise BootstrapError(f"remote artifact size mismatch: {relative}")
    return digest.hexdigest()


def verify_remote_completeness() -> dict[str, int]:
    lock = lean_runtime_manifest()
    source = Path(__file__).parent / "profiles" / "kaggle-l4x4-cu129" / "L4_INFERENCE_LOCK.json"
    if _sha256(source) != LOCK_SHA256:
        raise BootstrapError("packaged L4 inference lock SHA256 mismatch")
    rows = _remote_inventory()
    namespace = load_profile().data["public_artifacts"]["namespace"].strip("/")
    expected = [("metadata/L4_INFERENCE_LOCK.json", source.stat().st_size, LOCK_SHA256)]
    expected += [("wheelhouse/" + _safe_name(x["filename"]), x["size"], x["sha256"]) for x in lock["wheels"]]
    native = lock["native"]
    expected.append(("native/" + WHEEL_FILENAME, native["size"], native["sha256"]))
    missing = []
    mismatched = []
    for relative, size, digest in expected:
        row = rows.get(namespace + "/" + relative)
        if row is None:
            missing.append(relative)
        elif row.get("size") != size:
            mismatched.append(relative)
        else:
            lfs_digest = (row.get("lfs") or {}).get("oid")
            if lfs_digest is None and size <= 20 * 1024**2:
                lfs_digest = _remote_small_sha256(relative, size)
            if lfs_digest != digest:
                mismatched.append(relative)
    if missing or mismatched:
        raise BootstrapError(f"public runtime incomplete: missing={missing}; identity_mismatch={mismatched}")
    return {"artifact_count": len(expected), "download_bytes": sum(x[1] for x in expected)}


def fetch_public_runtime(cache_root: str | Path) -> Path:
    """Cache the exact 168 wheel selection; never resolve dependencies."""
    immutable_revision()
    root = _safe_root(_safe_root(Path(cache_root)) / "kaggle-vllm-v032-l4x4-lean-runtime")
    with _cache_lock(root):
        inventory = verify_remote_completeness()
        already = sum(p.stat().st_size for p in root.rglob("*.whl") if p.is_file() and not p.is_symlink())
        # Staging performs its own 24 GiB guard after the cache is complete.
        need = max(0, inventory["download_bytes"] - already) + 24 * 1024**3
        if shutil.disk_usage(root).free < need:
            raise BootstrapError(f"insufficient cache storage: need {need} free bytes before download")
        source = Path(__file__).parent / "profiles" / "kaggle-l4x4-cu129" / "L4_INFERENCE_LOCK.json"
        _download("metadata/L4_INFERENCE_LOCK.json", root / "metadata" / "RUNTIME_MANIFEST.json", source.stat().st_size, LOCK_SHA256)
        audit = source.parent / "TORCHCODEC_ELF_AUDIT.json"
        audit_target = root / "metadata" / audit.name
        if _sha256(audit) != TORCHCODEC_AUDIT_SHA256:
            raise BootstrapError("packaged TorchCodec audit identity mismatch")
        if not audit_target.exists():
            audit_target.write_bytes(audit.read_bytes())
        if audit_target.is_symlink() or _sha256(audit_target) != TORCHCODEC_AUDIT_SHA256:
            raise BootstrapError("cached TorchCodec audit identity mismatch")
        lock = lean_runtime_manifest()
        for record in lock["wheels"]:
            name = _safe_name(record["filename"])
            _download("wheelhouse/" + name, root / "wheelhouse" / name, record["size"], record["sha256"])
        native = lock["native"]
        wheel = root / "native" / WHEEL_FILENAME
        _download("native/" + WHEEL_FILENAME, wheel, native["size"], native["sha256"])
        validate_native_wheel(wheel)
        verify_runtime(root)
    return root
