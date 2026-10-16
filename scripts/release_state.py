"""Verify exact SDK artifacts and reuse an already published PyPI release.

Update release/sdk-exact-artifacts.json for each new reviewed SDK candidate.
The publish workflow still requires authenticated staging and attestation when
PyPI has no exact release. An identity collision fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "release" / "sdk-exact-artifacts.json"
NATIVE_LOCK = Path(__file__).resolve().parents[1] / "src" / "kaggle_vllm" / "profiles" / "kaggle-l4x4-cu129" / "L4_INFERENCE_LOCK.json"
GITHUB_RELEASE = "https://api.github.com/repos/kaggle-vllm/kaggle-vllm/releases/tags/v0.3.1"
HF_REPOSITORY = "waqasm86/kaggle-vllm-binaries"
HF_NATIVE_PATH = "v0.3.0/l4x4/native/vllm-0.30.0-cp312-cp312-linux_x86_64.whl"
GITHUB_CHECKSUMS = (299, "a6e770c2cb27791037d109870d9734bf121edc8f78223a0ac34a8fd265601d4d")


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


def _read_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "kaggle-vllm-release-state/0.3.1"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except http.client.RemoteDisconnected:
        # Some local proxies drop urllib's GitHub API connection. CI and the
        # local release audit both have curl; keep the same fixed HTTPS URL.
        response = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", url], capture_output=True, text=True, check=True, timeout=30)
        return json.loads(response.stdout)


def _native_identity() -> dict:
    return json.loads(NATIVE_LOCK.read_text())["native"]


def classify_github(data: dict, lock: dict, native: dict) -> str:
    if data.get("draft") or data.get("tag_name") != "v0.3.1":
        raise ValueError("GitHub v0.3.1 is not a published release")
    expected = {item["filename"]: (item["size"], item["sha256"]) for item in lock["files"]}
    expected[native["filename"]] = (native["size"], native["sha256"])
    expected["SHA256SUMS.txt"] = GITHUB_CHECKSUMS
    actual = {item["name"]: (item["size"], str(item.get("digest", "")).removeprefix("sha256:")) for item in data.get("assets", [])}
    if len(data.get("assets", [])) != len(actual) or set(actual) != set(expected):
        raise ValueError("GitHub release asset set differs from the exact published release")
    for name, identity in expected.items():
        if actual[name] != identity:
            raise ValueError(f"GitHub release artifact identity collision: {name}")
    return "GITHUB_RELEASE_COMPLETE"


def github_state(lock: dict) -> str:
    try:
        data = _read_json(GITHUB_RELEASE)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "GITHUB_RELEASE_PENDING"
        raise
    return classify_github(data, lock, _native_identity())


def classify_hf(data: dict, native: dict) -> str:
    files = {item.get("rfilename"): item for item in data.get("siblings", [])}
    item = files.get(HF_NATIVE_PATH)
    if item is None:
        return "HF_NATIVE_PENDING"
    lfs = item.get("lfs") or {}
    if lfs.get("sha256") != native["sha256"] or item.get("size") != native["size"] or lfs.get("size") != native["size"]:
        raise ValueError("Hugging Face native-wheel identity collision")
    revision = data.get("sha", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Hugging Face immutable revision unavailable")
    return f"HF_NATIVE_COMPLETE {revision}"


def hf_state() -> str:
    name = urllib.parse.quote(HF_REPOSITORY, safe="/")
    data = _read_json(f"https://huggingface.co/api/models/{name}?blobs=true")
    return classify_hf(data, _native_identity())


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
    parser.add_argument("command", choices=("pypi", "github", "hf-native", "public", "verify-dir"))
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
    elif args.command == "github":
        print(github_state(lock))
    elif args.command == "hf-native":
        print(hf_state())
    elif args.command == "public":
        print(pypi_state(lock))
        print(github_state(lock))
        print(hf_state())
    else:
        if args.directory is None:
            parser.error("verify-dir requires a directory")
        verify_dir(args.directory, lock)
        print("EXACT_DISTRIBUTIONS_VERIFIED")


if __name__ == "__main__":
    main()
