"""Write an exact, read-only additive Hub publication plan."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/v032-production-release"
STAGING = ROOT / "release/hf-v032-staging"
OUTPUT = ROOT / "release/hf-v032-upload-plan.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    inventory = json.loads((REPORT / "04-huggingface-existing-inventory.json").read_text())
    verified = json.loads((REPORT / "local-artifact-verification.json").read_text())
    if any(row["status"] != "PASS" for row in verified):
        raise ValueError("local artifacts are not all verified")
    lock = json.loads((REPORT / "07-v032-runtime-manifest.json").read_text())
    t4 = json.loads((REPORT / "06-v020-runtime-manifest.json").read_text())
    operations = []
    native_path = "v0.3.0/l4x4/native/" + lock["native"]["filename"]
    existing = next((row for row in inventory["files"] if row["path"] == native_path), None)
    if existing is None or existing.get("size") != lock["native"]["size"] or existing.get("sha256") != lock["native"]["sha256"]:
        raise ValueError("historical L4 wheel is not verified in the Hub inventory")
    local = {row["filename"]: row for row in verified}
    origin = json.loads((REPORT / "upstream-wheel-origins.json").read_text())
    upstream = {row["filename"]: row for row in origin["wheels"]}
    if len(upstream) != 168:
        raise ValueError("publisher source inventory is incomplete")
    for row in lock["wheels"]:
        actual, source = local[row["filename"]], upstream[row["filename"]]
        if actual["sha256"] != row["sha256"] or actual["size"] != row["size"]:
            raise ValueError(f"local lock mismatch: {row['filename']}")
        if source["sha256"] != row["sha256"] or source["size"] != row["size"] or source["status"] not in ("EXACT", "EXACT_DOWNLOADED"):
            raise ValueError(f"publisher source identity unverified: {row['filename']}")
    for path in sorted(STAGING.rglob("*")):
        if path.is_file():
            operations.append({"operation": "add", "local_path": str(path),
                               "target": path.relative_to(STAGING).as_posix(),
                               "size": path.stat().st_size,
                               "sha256": sha256(path)})
    OUTPUT.write_text(json.dumps({
        "schema_version": 1, "repository": "waqasm86/kaggle-vllm-binaries",
        "base_revision": inventory["revision"], "mode": "additive_only",
        "publication_authorization": "user-authorized-contingent-on-clearance-and-account-credentials",
        "requires_redistribution_clearance": True,
        "native_reference": {"path": native_path, "revision": inventory["revision"],
                             "sha256": lock["native"]["sha256"], "size": lock["native"]["size"]},
        "historical_t4_reference": {"path": t4["historical_path"],
                                     "revision": t4["immutable_revision"],
                                     "sha256": t4["native"]["sha256"]},
        "wheelhouse_delivery": "publisher-index; no dependency wheel uploads",
        "operations": operations,
    }, indent=2) + "\n")
    print(f"{len(operations)} planned operations; no upload performed")


if __name__ == "__main__":
    main()
