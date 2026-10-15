"""Audit license metadata in the exact P31/corrected-v2 runtime wheels.

This is a metadata inventory, not permission to redistribute any wheel.
Runs with local Python 3.11; never imports a native wheel.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import zipfile
from collections import Counter
from email import message_from_bytes
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOUSE = REPO.parent / "kaggle-vllm-v030-wheelhouse-cp312-cu129-v2"
P31 = REPO.parent / "kaggle-vllm-030-final-cuda-binaries-with-v2" / "p31-kaggle-vllm-v030-l4x4-final-v2.tar.gz"
PROFILE = REPO / "src/kaggle_vllm/profiles/kaggle-l4x4-cu129/RUNTIME_MANIFEST.json"
RATIONALE = REPO / "release/RUNTIME_PACKAGE_RATIONALE.json"
OUT = REPO / "release"
P31_SHA = "e82085d0c6f180d2f4f8eb8f52332a92051273715e0923082fc65fc8857ddad5"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wheel_metadata(archive: zipfile.ZipFile) -> dict:
    names = archive.namelist()
    metadata_files = [name for name in names if name.endswith(".dist-info/METADATA") and name.count("/") == 1]
    if len(metadata_files) != 1:
        raise ValueError("wheel must have exactly one dist-info/METADATA")
    metadata = message_from_bytes(archive.read(metadata_files[0]))
    licenses = sorted(name for name in names if ".dist-info/licenses/" in name or Path(name).name.lower() in ("license", "license.txt", "license.md", "copying", "notice", "notice.txt"))
    raw_license = metadata.get("License")
    expression = metadata.get("License-Expression")
    classifiers = sorted(x for x in metadata.get_all("Classifier", []) if x.startswith("License ::"))
    declared = bool(expression or (raw_license and raw_license.strip().upper() not in ("UNKNOWN", "NONE")) or classifiers)
    return {
        "metadata_name": metadata.get("Name"),
        "metadata_version": metadata.get("Version"),
        "license_expression": expression,
        "license_field": raw_license[:1000] if raw_license else None,
        "license_field_truncated": bool(raw_license and len(raw_license) > 1000),
        "license_classifiers": classifiers,
        "embedded_license_or_notice_files": licenses,
        "license_metadata_status": "DECLARED" if declared else "UNCLEAR",
        "redistribution_status": "MANUAL_REVIEW",
        "redistribution_note": "Wheel metadata does not establish redistribution clearance; review upstream terms and carry required notices before uploading.",
    }


def main() -> None:
    manifest = json.loads(PROFILE.read_text())
    rationale = json.loads(RATIONALE.read_text())
    paths = {item["distribution"]: item for item in rationale["packages"]}
    records: list[dict] = []
    for wheel in manifest["wheels"]:
        path = HOUSE / wheel["group"] / wheel["filename"]
        if not path.is_file() or path.stat().st_size != wheel["size"] or digest(path) != wheel["sha256"]:
            raise RuntimeError(f"source wheel identity mismatch: {path}")
        with zipfile.ZipFile(path) as archive:
            try:
                item = wheel_metadata(archive)
            except ValueError as error:
                raise ValueError(f"{path}: {error}") from error
        normalized_name = re.sub(r"[-_.]+", "-", item["metadata_name"] or "").lower()
        if normalized_name != wheel["distribution"] or item["metadata_version"] != wheel["version"]:
            raise RuntimeError(f"wheel METADATA identity mismatch: {path}")
        reason = paths[wheel["distribution"]]
        records.append({"distribution": wheel["distribution"], "version": wheel["version"], "source_wheel_filename": wheel["filename"], "source_wheel_sha256": wheel["sha256"], "source_group": wheel["group"], "dependency_root": reason["root"], "dependency_parent_path": reason["root_path"], "active_requires_dist": reason["active_requires_dist"], **item})
    if P31.stat().st_size != 291899459 or digest(P31) != P31_SHA:
        raise RuntimeError("P31 final-v2 identity mismatch")
    with tarfile.open(P31, "r:gz") as tar:
        selected = [member for member in tar if Path(member.name).name == manifest["native"]["filename"]]
        if len(selected) != 1 or not selected[0].isfile() or selected[0].size != manifest["native"]["size"]:
            raise RuntimeError("P31 native member identity mismatch")
        native = tar.extractfile(selected[0]).read()
    if hashlib.sha256(native).hexdigest() != manifest["native"]["sha256"]:
        raise RuntimeError("P31 native wheel hash mismatch")
    with zipfile.ZipFile(io.BytesIO(native)) as archive:
        item = wheel_metadata(archive)
    records.append({"distribution": "vllm", "version": "0.30.0", "source_wheel_filename": manifest["native"]["filename"], "source_wheel_sha256": manifest["native"]["sha256"], "source_group": "native-p31-final-v2", "dependency_root": "vllm", "dependency_parent_path": ["vllm"], "active_requires_dist": [], **item})
    if len(records) != 196:
        raise RuntimeError("runtime wheel count changed")
    records.sort(key=lambda item: item["distribution"])
    summary = Counter(item["license_metadata_status"] for item in records)
    output = {"schema_version": 1, "release": "kaggle-vllm 0.3.0 candidate", "runtime_manifest_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(), "p31_final_v2_sha256": P31_SHA, "wheel_count": len(records), "declared_license_metadata_count": summary["DECLARED"], "unclear_license_metadata_count": summary["UNCLEAR"], "redistribution_clearance": "NOT_GRANTED_BY_THIS_AUDIT", "manual_review_required_count": len(records), "packages": records}
    (OUT / "v0.3.0-license-inventory.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = ["# kaggle-vllm 0.3.0 third-party notices preparation", "", "This inventory reads the exact P31 native wheel and corrected v2 dependency wheels. It does **not** grant redistribution permission. Review every upstream license, notice, and binary redistribution condition before publicly uploading runtime-v3 or standalone wheels.", "", "The SDK wraps upstream vLLM 0.30.0; it is not a fork. Preserve vLLM's Apache-2.0 attribution and bundled notices. Model licenses are separate: Gemma 4 is a mounted Kaggle model and is not included in runtime-v3. The competition acceptance runs no Qwen model; any later permitted Qwen test must handle its license and notices separately.", "", f"Wheels: {len(records)}. Declared license metadata: {summary['DECLARED']}. Unclear metadata: {summary['UNCLEAR']}. Redistribution status for every wheel: MANUAL_REVIEW.", "", "| Distribution | Version | License metadata | Embedded license/notice files | Redistribution |", "| --- | --- | --- | ---: | --- |"]
    for row in records:
        label = row["license_expression"] or row["license_field"] or (", ".join(row["license_classifiers"]) if row["license_classifiers"] else "UNCLEAR")
        label = label.replace("|", "/").replace("\n", " ")[:120]
        lines.append(f"| {row['distribution']} | {row['version']} | {label} | {len(row['embedded_license_or_notice_files'])} | MANUAL_REVIEW |")
    lines.extend(["", "The machine-readable [license inventory](v0.3.0-license-inventory.json) names each source wheel, its SHA256, declared metadata, embedded license files, and dependency-parent path. A future publication review must obtain/retain full required license texts and notices; this document is not that approval.", ""])
    (OUT / "v0.3.0-third-party-notices.md").write_text("\n".join(lines))
    print(json.dumps({"wheel_count": len(records), "declared": summary["DECLARED"], "unclear": summary["UNCLEAR"], "manual_review": len(records)}))


if __name__ == "__main__":
    main()
