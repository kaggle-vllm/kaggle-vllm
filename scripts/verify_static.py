"""Verify identities shared by profiles, provenance, and checksum manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import tomllib
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "src/kaggle_vllm/profiles/kaggle-t4x2-cu128/profile.json"
COMPAT = ROOT / "compat/kaggle-t4x2-cu128.json"
PROVENANCE = ROOT / "artifacts/BUILD-PROVENANCE.json"
CHECKSUMS = ROOT / "artifacts/SHA256SUMS.txt"


def main() -> int:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    compat = json.loads(COMPAT.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    checksums = CHECKSUMS.read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    wheel_sha = profile["wheel"]["sha256"]
    source_commit = profile["vllm"]["source_commit"]
    assert compat["wheel_sha256"] == wheel_sha
    assert provenance["native_wheel"]["sha256"] == wheel_sha
    assert compat["upstream_vllm_commit"] == source_commit
    assert provenance["upstream"]["commit"] == source_commit
    assert wheel_sha in checksums
    assert compat["driver_reported_cuda_max"] == "13.0"
    forbidden = {"torch", "torchvision", "torchaudio", "vllm"}
    dependencies = {
        Requirement(item).name.casefold() for item in project.get("dependencies", [])
    }
    assert not forbidden & dependencies
    print("static profile/provenance/package checks: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
