#!/usr/bin/python3.12
"""Stage the exact v0.3.2 Kaggle wheels and optionally install globally.

This script makes no network requests. Without --apply it only verifies and
stages files. --apply also requires --replace-global-cuda-stack because pip
will replace Kaggle's preinstalled Torch/CUDA packages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ARCHIVES = {
    "runtime-core": "kaggle-vllm-v030-py312-runtime-core-v2.zip",
    "runtime-large": "kaggle-vllm-v030-py312-runtime-large-v2.zip",
    "torch-cu129": "kaggle-vllm-v030-py312-torch-cu129-v2.zip",
}
LOCK_MEMBER = "kaggle_vllm/profiles/kaggle-l4x4-cu129/L4_INFERENCE_LOCK.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_file(path: Path, record: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing or symlinked: {path}")
    if path.name != record["filename"]:
        raise ValueError(f"filename mismatch: {path.name} != {record['filename']}")
    if path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
        raise ValueError(f"size or SHA256 mismatch: {path}")


def safe_stage(root: Path, release_sha: str) -> Path:
    root = root.expanduser().absolute()
    if root.resolve(strict=False) != root or root.is_symlink():
        raise ValueError(f"stage path traverses a symlink: {root}")
    if root.exists() and not root.is_dir():
        raise ValueError(f"stage root is not a directory: {root}")
    marker = root / "release-manifest.sha256"
    if root.exists() and any(root.iterdir()):
        if marker.is_symlink() or not marker.is_file() or marker.read_text().strip() != release_sha:
            raise ValueError(f"stage root is nonempty and has another or unknown owner: {root}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text(release_sha + "\n", encoding="ascii")
    wheelhouse = root / "wheelhouse"
    if wheelhouse.is_symlink():
        raise ValueError("wheelhouse cannot be a symlink")
    wheelhouse.mkdir(exist_ok=True)
    return wheelhouse


def stage_member(archive: zipfile.ZipFile, name: str, record: dict, output: Path) -> None:
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError(f"unsafe ZIP member: {name}")
    info = archive.getinfo(name)
    if info.is_dir() or info.file_size != record["size"] or ((info.external_attr >> 16) & 0o170000) == 0o120000:
        raise ValueError(f"unsafe or wrong-sized ZIP member: {name}")
    if output.exists():
        check_file(output, record)
        return
    if output.is_symlink():
        raise ValueError(f"wheel target is a symlink: {output}")
    with tempfile.NamedTemporaryFile(prefix=".wheel-", dir=output.parent, delete=False) as target:
        temporary = Path(target.name)
        h = hashlib.sha256()
        count = 0
        try:
            with archive.open(info) as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    count += len(block)
                    if count > record["size"]:
                        raise ValueError(f"ZIP member exceeds pinned size: {name}")
                    target.write(block)
                    h.update(block)
            target.flush()
            os.fsync(target.fileno())
            if count != record["size"] or h.hexdigest() != record["sha256"]:
                raise ValueError(f"ZIP wheel hash mismatch: {name}")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path, help="mounted Kaggle Input containing the pinned v2 ZIPs and native wheel")
    parser.add_argument("--sdk-wheel", required=True, type=Path)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument("--stage-root", type=Path, default=Path("/tmp/kaggle-vllm/v0.3.2/global-install"))
    parser.add_argument("--apply", action="store_true", help="run pip after all identities pass")
    parser.add_argument("--replace-global-cuda-stack", action="store_true", help="explicitly allow replacement of Kaggle's preinstalled Torch/CUDA packages")
    args = parser.parse_args()
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12) or Path(sys.executable).resolve() != Path("/usr/bin/python3.12"):
        parser.error("run exactly /usr/bin/python3.12 (CPython 3.12)")
    if sys.prefix != sys.base_prefix:
        parser.error("global installation requires the ordinary Python 3.12 environment")
    if args.apply and not args.replace_global_cuda_stack:
        parser.error("--apply requires --replace-global-cuda-stack")
    input_root = args.input_root.expanduser().resolve(strict=True)
    if not input_root.is_dir():
        parser.error("--input-root must be a directory")
    manifest_path = args.release_manifest.expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PENDING_KAGGLE_ACCEPTANCE" or manifest.get("sdk_version") != "0.3.2":
        raise ValueError("wrong or incomplete v0.3.2 release manifest")
    check_file(args.sdk_wheel, manifest["sdk_wheel"])
    native = input_root / manifest["native_wheel"]["filename"]
    check_file(native, manifest["native_wheel"])
    with zipfile.ZipFile(args.sdk_wheel) as sdk:
        lock_bytes = sdk.read(LOCK_MEMBER)
    if hashlib.sha256(lock_bytes).hexdigest() != manifest["dependency_lock_sha256"]:
        raise ValueError("SDK wheel embeds a different dependency lock")
    lock = json.loads(lock_bytes)
    if lock.get("wheel_count") != 168 or len(lock.get("wheels", [])) != 168 or lock.get("native") != manifest["native_wheel"]:
        raise ValueError("SDK wheel has an unexpected P31 inference lock")
    release_sha = digest(manifest_path)
    wheelhouse = safe_stage(args.stage_root, release_sha)
    needed = sum(r["size"] for r in lock["wheels"]) + manifest["sdk_dependency"]["size"]
    if shutil.disk_usage(wheelhouse).free < needed + 10 * 1024**3:
        raise ValueError("insufficient free disk for verified wheel staging and global pip installation")
    groups: dict[str, list[dict]] = {name: [] for name in ARCHIVES}
    for record in lock["wheels"]:
        if record["group"] not in groups:
            raise ValueError(f"unexpected wheel group: {record['group']}")
        groups[record["group"]].append(record)
    for group, filename in ARCHIVES.items():
        archive_record = manifest["archives"][filename]
        archive = input_root / filename
        check_file(archive, archive_record)
        with zipfile.ZipFile(archive) as source:
            for record in groups[group]:
                stage_member(source, record["filename"], record, wheelhouse / record["filename"])
    build_archive_name = manifest["sdk_dependency_archive"]["filename"]
    build_archive = input_root / build_archive_name
    check_file(build_archive, manifest["sdk_dependency_archive"])
    packaging_record = manifest["sdk_dependency"]
    with zipfile.ZipFile(build_archive) as source:
        stage_member(source, packaging_record["filename"], packaging_record, wheelhouse / packaging_record["filename"])
    requirements = [*lock["wheels"], packaging_record, lock["native"], manifest["sdk_wheel"]]
    names = [r["filename"] for r in requirements]
    if len(set(names)) != len(names):
        raise ValueError("release lock contains duplicate wheel filenames")
    lines = []
    for record in requirements:
        if record is lock["native"]:
            distribution, version = "vllm", "0.30.0"
        else:
            distribution, version = record["distribution"], record["version"]
        lines.append(f"{distribution}=={version} --hash=sha256:{record['sha256']}")
    target = wheelhouse.parent / "requirements-locked.txt"
    if target.is_symlink():
        raise ValueError("requirements target cannot be a symlink")
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(json.dumps({"status": "VERIFIED_STAGED", "wheel_count": len(requirements), "stage": str(wheelhouse), "apply": args.apply}, indent=2))
    if not args.apply:
        return 0
    command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-cache-dir", "--no-compile", "--force-reinstall", "--only-binary=:all:", "--require-hashes", "--find-links", str(wheelhouse), "--find-links", str(native.parent), "--find-links", str(args.sdk_wheel.parent), "-r", str(target)]
    environment = dict(os.environ, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    subprocess.run(command, check=True, env=environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
