"""Pinned, checksum-verified Hugging Face artifact delivery."""

from __future__ import annotations

import importlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import quote
from urllib.request import Request, urlopen

from .checksums import verify_sha256
from .exceptions import ChecksumMismatchError, DownloadError
from .profiles import BootstrapProfile


class _Response(Protocol):
    def __enter__(self) -> BinaryIO: ...

    def __exit__(self, *args: object) -> None: ...


UrlOpener = Callable[..., _Response]


def immutable_resolve_url(profile: BootstrapProfile) -> str:
    """Build the immutable Hugging Face ``resolve`` fallback URL."""

    repo = quote(profile.hf_repo_id, safe="/")
    revision = quote(profile.hf_revision, safe="")
    filename = quote(profile.wheel_filename, safe="")
    return f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _hub_downloader() -> Callable[..., str] | None:
    try:
        module = importlib.import_module("huggingface_hub")
    except (ImportError, OSError):
        return None
    function = getattr(module, "hf_hub_download", None)
    return function if callable(function) else None


def _download_with_https(
    profile: BootstrapProfile,
    destination: Path,
    *,
    opener: UrlOpener = urlopen,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        immutable_resolve_url(profile),
        headers={"User-Agent": "kaggle-vllm/0.1 bootstrap"},
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            with opener(request) as response:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
        verify_sha256(temporary, profile.wheel_sha256)
        os.replace(temporary, destination)
    except ChecksumMismatchError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    except (OSError, ValueError) as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise DownloadError(f"HTTPS wheel download failed: {error}") from error
    return destination


def download_wheel(
    profile: BootstrapProfile,
    cache_dir: str | Path,
    *,
    prefer_hub: bool = True,
    opener: UrlOpener = urlopen,
) -> Path:
    """Download the exact profiled wheel and require its expected SHA256.

    A valid existing destination is reused. When available, ``hf_hub_download``
    supplies normal Hub cache/Xet behavior. Otherwise a standard-library HTTPS
    request follows Hugging Face redirects and writes atomically.
    """

    cache = Path(cache_dir).expanduser().resolve()
    destination = cache / profile.wheel_filename
    if destination.is_file():
        verify_sha256(destination, profile.wheel_sha256)
        return destination
    if destination.exists():
        raise DownloadError(f"wheel cache destination is not a file: {destination}")

    if prefer_hub and (hub_download := _hub_downloader()) is not None:
        try:
            downloaded = Path(
                hub_download(
                    repo_id=profile.hf_repo_id,
                    filename=profile.wheel_filename,
                    revision=profile.hf_revision,
                    repo_type="model",
                    local_dir=str(cache),
                )
            ).resolve()
            verify_sha256(downloaded, profile.wheel_sha256)
            return downloaded
        except ChecksumMismatchError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise DownloadError(
                f"Hugging Face Hub wheel download failed: {error}"
            ) from error

    return _download_with_https(profile, destination, opener=opener)
