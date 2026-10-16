"""Normalize only archive metadata in a locally built 0.3.2 sdist."""
from __future__ import annotations

import argparse
import gzip
import os
import tarfile
import tempfile
from pathlib import Path


def normalize(source: Path, destination: Path, epoch: int) -> None:
    if source.name != "kaggle_vllm-0.3.2.tar.gz" or destination.name != source.name:
        raise ValueError("only the 0.3.2 source distribution is supported")
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".sdist-", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with tarfile.open(source, "r:gz") as src, temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch, compresslevel=9) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as out:
                    for member in src:
                        if not (member.isfile() or member.isdir()) or not member.name.startswith("kaggle_vllm-0.3.2/") and member.name != "kaggle_vllm-0.3.2":
                            raise ValueError(f"unsafe sdist member: {member.name}")
                        member.uid = member.gid = 0
                        member.uname = member.gname = ""
                        member.mtime = epoch
                        member.pax_headers = {}
                        stream = src.extractfile(member) if member.isfile() else None
                        out.addfile(member, stream)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    normalize(args.source, args.destination, args.epoch)


if __name__ == "__main__":
    main()
