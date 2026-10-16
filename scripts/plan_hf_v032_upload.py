"""Write an exact, read-only additive Hub publication plan."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/v032-production-release"
STAGING = ROOT / "release/hf-v032-staging"
OUTPUT = ROOT / "release/hf-v032-upload-plan.json"


def main() -> None:
    inventory = json.loads((REPORT / "04-huggingface-existing-inventory.json").read_text())
    verified = json.loads((REPORT / "local-artifact-verification.json").read_text())
    if any(row["status"] != "PASS" for row in verified):
        raise ValueError("local artifacts are not all verified")
    lock = json.loads((REPORT / "07-v032-runtime-manifest.json").read_text())
    t4 = json.loads((REPORT / "06-v020-runtime-manifest.json").read_text())
    operations = [
        {"operation": "copy", "source": t4["historical_path"],
         "source_revision": t4["immutable_revision"],
         "target": "v0.2.0/native/" + t4["native"]["filename"],
         "sha256": t4["native"]["sha256"], "size": t4["native"]["size"]},
        {"operation": "copy", "source": "v0.3.0/l4x4/native/" + lock["native"]["filename"],
         "source_revision": inventory["revision"],
         "target": "v0.3.2/l4x4/native/" + lock["native"]["filename"],
         "sha256": lock["native"]["sha256"], "size": lock["native"]["size"]},
    ]
    local = {row["filename"]: row for row in verified}
    for row in lock["wheels"]:
        actual = local[row["filename"]]
        if actual["sha256"] != row["sha256"] or actual["size"] != row["size"]:
            raise ValueError(f"local lock mismatch: {row['filename']}")
        operations.append({
            "operation": "add", "local_path": actual["path"],
            "target": "v0.3.2/l4x4/wheelhouse/" + row["filename"],
            "sha256": row["sha256"], "size": row["size"],
            "license_status": "REVIEW_REQUIRED",
        })
    for path in sorted(STAGING.rglob("*")):
        if path.is_file():
            operations.append({"operation": "add", "local_path": str(path),
                               "target": path.relative_to(STAGING).as_posix(),
                               "size": path.stat().st_size})
    OUTPUT.write_text(json.dumps({
        "schema_version": 1, "repository": "waqasm86/kaggle-vllm-binaries",
        "base_revision": inventory["revision"], "mode": "additive_only",
        "requires_explicit_user_authorization": True,
        "requires_redistribution_clearance": True,
        "operations": operations,
    }, indent=2) + "\n")
    print(f"{len(operations)} planned operations; no upload performed")


if __name__ == "__main__":
    main()
