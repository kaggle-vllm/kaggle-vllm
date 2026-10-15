"""Verify exact SDK artifacts and reuse an already published PyPI release.

Update release/sdk-exact-artifacts.json for each new reviewed SDK candidate.
The publish workflow still requires authenticated staging and attestation when
PyPI has no exact release. An identity collision fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "release" / "sdk-exact-artifacts.json"


def load_lock(path: Path = LOCK) -> dict:
    data = json.loads(path.read_text())
    if not data.get("project") or not data.get("version"):
        raise ValueError("missing SDK project or version in release lock")
    files = data["files"]
    if len(files) != 2 or len({item["filename"] for item in files}) != 2:
        raise ValueError("release lock requires two distinct distributions")
    for item in files:
        if Path(item["filename"]).name != item["filename"]:
            raise ValueError("release lock filenames must be basenames")
        if item["size"] <= 0 or len(item["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in item["sha256"]):
            raise ValueError("invalid release lock size or SHA256")
    return data


def classify_pypi(data: dict, lock: dict) -> str:
    if data.get("info", {}).get("version") != lock["version"]:
        raise ValueError("PyPI version identity mismatch")
    expected = {item["filename"]: (item["size"], item["sha256"]) for item in lock["files"]}
    actual = {item["filename"]: (item["size"], item["digests"]["sha256"]) for item in data["urls"]}
    if actual != expected:
        raise ValueError(f"PyPI artifact identity collision for {lock['project']} {lock['version']}; stop")
    return "SDK_RELEASE_COMPLETE"


def pypi_state(lock: dict) -> str:
    project = urllib.parse.quote(lock["project"], safe="")
    version = urllib.parse.quote(lock["version"], safe="")
    url = f"https://pypi.org/pypi/{project}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "SDK_RELEASE_PENDING"
        raise
    return classify_pypi(data, lock)


def verify_dir(root: Path, lock: dict) -> None:
    expected = {item["filename"]: item for item in lock["files"]}
    actual = {p.name for p in root.iterdir()}
    if actual != set(expected) or any(not p.is_file() or p.is_symlink() for p in root.iterdir()):
        raise ValueError(f"distribution filename set mismatch: {sorted(actual)}")
    for name, item in expected.items():
        path = root / name
        if path.stat().st_size != item["size"]:
            raise ValueError(f"distribution size mismatch: {name}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"distribution SHA256 mismatch: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("pypi", "verify-dir"))
    parser.add_argument("directory", nargs="?", type=Path)
    args = parser.parse_args()
    lock = load_lock()
    if args.command == "pypi":
        result = pypi_state(lock)
        print(result)
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a", encoding="utf-8") as stream:
                stream.write(f"published={'true' if result == 'SDK_RELEASE_COMPLETE' else 'false'}\n")
    else:
        if args.directory is None:
            parser.error("verify-dir requires a directory")
        verify_dir(args.directory, lock)
        print("EXACT_DISTRIBUTIONS_VERIFIED")


if __name__ == "__main__":
    main()
