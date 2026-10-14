"""Prepare, but never upload, the three-file P31 native-wheel Kaggle Dataset.

Run locally with the custom python3.11. The Kaggle CLI metadata control file is
created later by `kaggle datasets init`; this helper never uses credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_P31 = ROOT.parent / "kaggle-vllm-030-final-cuda-binaries-with-v2" / "p31-kaggle-vllm-v030-l4x4-final-v2.tar.gz"
DEFAULT_OUTPUT = ROOT / "dist" / "kaggle-vllm-v030-native-wheel-only"
WHEEL = "vllm-0.30.0-cp312-cp312-linux_x86_64.whl"
P31_SHA = "e82085d0c6f180d2f4f8eb8f52332a92051273715e0923082fc65fc8857ddad5"
WHEEL_SHA = "9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8"
MEMBER = "p31-kaggle-vllm-v030-l4x4-final-v2/" + WHEEL


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--p31-archive", type=Path, default=DEFAULT_P31)
    choice.add_argument("--native-wheel", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing existing output: {args.output}")
    source = args.native_wheel or args.p31_archive
    if args.native_wheel:
        if source.name != WHEEL or source.stat().st_size != 293855397 or sha256(source) != WHEEL_SHA:
            raise SystemExit("native P31 wheel filename, size or SHA256 mismatch")
    else:
        if source.name != DEFAULT_P31.name or source.stat().st_size != 291899459 or sha256(source) != P31_SHA:
            raise SystemExit("P31 final-v2 filename, size or SHA256 mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="native-wheel-only-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / args.output.name
        stage.mkdir()
        target = stage / WHEEL
        if args.native_wheel:
            shutil.copyfile(source, target)
        else:
            with tarfile.open(source, "r:gz") as archive:
                members = [item for item in archive if item.name == MEMBER]
                if len(members) != 1 or not members[0].isfile() or members[0].size != 293855397:
                    raise SystemExit("P31 tar lacks the one exact regular native wheel member")
                member = archive.extractfile(members[0])
                assert member is not None
                with target.open("wb") as output:
                    shutil.copyfileobj(member, output)
        if target.stat().st_size != 293855397 or sha256(target) != WHEEL_SHA:
            raise SystemExit("extracted native wheel identity mismatch")
        with zipfile.ZipFile(target) as wheel_zip:
            names = set(wheel_zip.namelist())
            if wheel_zip.testzip() is not None or not {"vllm/_C_stable_libtorch.abi3.so", "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"} <= names:
                raise SystemExit("native wheel ZIP structure/core/FA2 mismatch")
        (stage / "SHA256SUMS.txt").write_text(f"{WHEEL_SHA}  {WHEEL}\n")
        identity = {"schema_version": 1, "release": "kaggle-vllm 0.3.0 candidate", "source": "P31 final-v2", "p31_archive_sha256": P31_SHA, "wheel_filename": WHEEL, "wheel_size": 293855397, "wheel_sha256": WHEEL_SHA, "native_runtime": "CPython 3.12/cp312; Linux x86_64; four NVIDIA L4/SM89"}
        (stage / "SOURCE_IDENTITY.json").write_text(json.dumps(identity, indent=2) + "\n")
        if len(list(stage.iterdir())) != 3:
            raise SystemExit("native Dataset payload must contain exactly three data files")
        os.replace(stage, args.output)
    print(f"Prepared {args.output} with three verified data files; no upload performed.")
    if shutil.which("kaggle"):
        print(f"Optional later private Kaggle CLI path: kaggle datasets init -p '{args.output}'")
        print("Review the generated datasets-metadata.json control file and its private visibility before running any create command.")
        print(f"Later, only when authorized: kaggle datasets create -p '{args.output}'")
    else:
        print("Kaggle CLI unavailable. Upload the three data files through the Kaggle Dataset UI later, or install/configure the CLI separately; this helper never uploads.")


if __name__ == "__main__":
    main()
