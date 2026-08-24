"""Streaming SHA256 helpers for large binary artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .exceptions import ChecksumMismatchError


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate a file SHA256 without loading it into memory."""

    artifact = Path(path)
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str | Path, expected: str) -> str:
    """Return the normalized digest or raise on mismatch."""

    normalized = expected.strip().split()[0].casefold()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError("expected SHA256 must contain exactly 64 hexadecimal characters")
    actual = sha256_file(path)
    if actual != normalized:
        raise ChecksumMismatchError(
            f"SHA256 mismatch for {Path(path)}: expected {normalized}, got {actual}"
        )
    return actual
