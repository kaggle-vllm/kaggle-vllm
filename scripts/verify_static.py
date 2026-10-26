"""Verify identities shared by profiles, provenance, and checksum manifests."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import tomllib
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "src/kaggle_vllm/profiles/kaggle-t4x2-cu128/profile.json"
COMPAT = ROOT / "compat/kaggle-t4x2-cu128.json"
PROVENANCE = ROOT / "artifacts/BUILD-PROVENANCE.json"
CHECKSUMS = ROOT / "artifacts/SHA256SUMS.txt"
INIT = ROOT / "src/kaggle_vllm/__init__.py"


def main() -> int:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    compat = json.loads(COMPAT.read_text(encoding="utf-8"))
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    checksums = CHECKSUMS.read_text(encoding="utf-8")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    historical = json.loads((ROOT / "compat/historical-py312-v0.2.0/profile.json").read_text(encoding="utf-8"))
    wheel_sha = historical["wheel"]["sha256"]
    source_commit = profile["vllm"]["source_commit"]
    assert profile["status"] == "target_unvalidated"
    assert profile["python"]["abi"] == "cp313"
    assert profile["wheel"]["filename"] is None
    assert profile["wheel"]["sha256"] is None
    assert compat["wheel_sha256"] is None
    assert provenance["native_wheel"]["sha256"] == wheel_sha
    assert compat["upstream_vllm_commit"] == source_commit
    assert provenance["upstream"]["commit"] == source_commit
    assert wheel_sha in checksums
    assert compat["status"] == "target_unvalidated"
    assert compat["gpus"] == 2
    assert compat["compute_capability"] == "7.5"
    assert project["requires-python"] == ">=3.13,<3.14"
    forbidden = {"torch", "torchvision", "torchaudio", "vllm"}
    dependencies = {
        Requirement(item).name.casefold() for item in project.get("dependencies", [])
    }
    assert not forbidden & dependencies
    version_match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
        INIT.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert version_match is not None
    assert version_match.group(1) == project["version"]
    active_files = [ROOT / "pyproject.toml", COMPAT]
    active_files.extend((ROOT / "src/kaggle_vllm").rglob("*.py"))
    active_files.extend((ROOT / "src/kaggle_vllm/profiles").rglob("*.json"))
    active_files.extend((ROOT / "kaggle-notebooks").rglob("*.ipynb"))
    # Assemble the later-generation markers so the scan itself does not
    # introduce a false match into active source text.
    forbidden_markers = (
        "0." + "3.0", "NVIDIA " + "L4", "SM" + "89", "sm_" + "89",
        "TP=" + "4", "cu" + "129", "CUDA 12." + "9",
    )
    for path in active_files:
        content = path.read_text(encoding="utf-8")
        assert not any(marker in content for marker in forbidden_markers), path
    print("static profile/provenance/package checks: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
