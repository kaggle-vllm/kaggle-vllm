"""Content hashing and fail-closed provenance verification."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from .errors import ProvenanceError


def sha256_file(path: str | Path) -> str:
    """Calculate a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256_manifest(root: str | Path, manifest: str | Path) -> dict[str, str]:
    """Verify a two-column SHA256SUMS manifest without path traversal."""

    base = Path(root).resolve()
    manifest_path = Path(manifest)
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ProvenanceError(f"cannot read checksum manifest: {manifest_path}") from error
    if not lines:
        raise ProvenanceError("checksum manifest is empty")
    verified: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ProvenanceError(f"malformed checksum line {number}")
        expected, raw_name = parts
        name = raw_name.lstrip("* ")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in verified:
            raise ProvenanceError(f"unsafe or duplicate checksum path: {name}")
        target = (base / Path(*relative.parts)).resolve()
        if base not in target.parents:
            raise ProvenanceError(f"checksum path escapes evidence root: {name}")
        if not target.is_file():
            raise ProvenanceError(f"checksummed file is absent: {name}")
        actual = sha256_file(target)
        if actual != expected.casefold():
            raise ProvenanceError(f"SHA256 mismatch for {name}")
        verified[name] = actual
    return verified
