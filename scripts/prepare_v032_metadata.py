"""Prepare local HF metadata and legal inventory. Performs no upload."""
from __future__ import annotations

import email
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/v032-production-release"
STAGING = ROOT / "release/hf-v032-staging"
L4_PROFILE = ROOT / "src/kaggle_vllm/profiles/kaggle-l4x4-cu129"
T4_PROFILE = ROOT / "src/kaggle_vllm/profiles/kaggle-t4x2-cu128"
WHEELHOUSE = ROOT.parent / "kaggle-vllm-v030-wheelhouse-cp312-cu129-v2"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    l4_lock_path = L4_PROFILE / "L4_INFERENCE_LOCK.json"
    l4_lock = json.loads(l4_lock_path.read_text())
    t4 = json.loads((T4_PROFILE / "profile.json").read_text())
    t4_manifest = {
        "schema_version": 1, "sdk_version": "0.2.0", "profile": t4["name"],
        "python_abi": t4["python"]["abi"], "cuda": t4["runtime"]["torch_cuda"],
        "torch": t4["runtime"]["torch"], "gpu": "2 x NVIDIA Tesla T4 / SM75", "tensor_parallel": 2,
        "native": {"filename": t4["wheel"]["filename"], "sha256": t4["wheel"]["sha256"], "size": 383922260},
        "hf_repo_id": t4["wheel"]["hf_repo_id"], "immutable_revision": t4["wheel"]["hf_revision"],
        "historical_path": t4["wheel"]["filename"], "new_path_status": "PREPARING",
    }
    write_json(REPORT / "06-v020-runtime-manifest.json", t4_manifest)
    write_json(STAGING / "v0.2.0/metadata/runtime-manifest.json", t4_manifest)
    (REPORT / "07-v032-runtime-manifest.json").write_bytes(l4_lock_path.read_bytes())
    meta = STAGING / "v0.3.2/l4x4/metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "L4_INFERENCE_LOCK.json").write_bytes(l4_lock_path.read_bytes())
    (meta / "runtime-manifest.json").write_bytes(l4_lock_path.read_bytes())
    (meta / "TORCHCODEC_ELF_AUDIT.json").write_bytes((L4_PROFILE / "TORCHCODEC_ELF_AUDIT.json").read_bytes())
    (meta / "UPSTREAM_WHEEL_SOURCES.json").write_bytes((L4_PROFILE / "UPSTREAM_WHEEL_SOURCES.json").read_bytes())
    write_json(meta / "compatibility.json", {
        "sdk": "0.3.2", "profile": "kaggle-l4x4-cu129", "python_abi": "cp312",
        "platform": "linux_x86_64", "gpu": "exactly 4 NVIDIA L4 / SM89", "tensor_parallel": 4,
        "torch": "2.13.0+cu129", "cuda_user_space": "12.9", "vllm": "0.30.0",
        "wheelhouse_delivery": "publisher-index", "wheel_count": 168,
        "gemma4_tested_backend_historical": "TRITON_ATTN", "manual_032_acceptance": "NOT RUN",
    })
    write_json(meta / "provenance.json", {
        "native_source_tag": "v0.30.0", "native_source_commit": "ced6857afa0ea7b2e3f0846a62e1394e90f15607",
        "native_filename": l4_lock["native"]["filename"], "native_sha256": l4_lock["native"]["sha256"],
        "wheel_lock_sha256": sha(l4_lock_path), "qualification": "historical P33 candidate only",
        "new_sdk_acceptance": "NOT RUN",
    })
    write_json(STAGING / "v0.2.0/metadata/provenance.json", {
        "native_source_tag": "v0.18.1", "native_source_commit": t4["vllm"]["source_commit"],
        "wheel_sha256": t4["wheel"]["sha256"], "historical_revision": t4["wheel"]["hf_revision"],
        "historic_path_preserved": True,
    })
    sums = [f"{l4_lock['native']['sha256']}  native/{l4_lock['native']['filename']}"]
    # Wheels are fetched from their original publisher indexes with pinned SHA256;
    # they are not staged for a third-party public mirror.
    sums += [f"{sha(meta / name)}  metadata/{name}" for name in
             ("L4_INFERENCE_LOCK.json", "runtime-manifest.json", "TORCHCODEC_ELF_AUDIT.json", "UPSTREAM_WHEEL_SOURCES.json", "compatibility.json", "provenance.json")]
    (meta / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n")
    (STAGING / "v0.2.0/metadata/SHA256SUMS.txt").write_text(
        f"{t4['wheel']['sha256']}  native/{t4['wheel']['filename']}\n"
        f"{sha(STAGING / 'v0.2.0/metadata/runtime-manifest.json')}  metadata/runtime-manifest.json\n"
    )
    releases = {
        "schema_version": 1, "repository": "waqasm86/kaggle-vllm-binaries",
        "releases": [
            {"sdk_version": "0.2.0", "gpu_profile": "T4x2/SM75", "vllm": "0.18.1 source", "python_abi": "cp312",
             "cuda": "12.8", "torch": "2.10.0+cu128", "manifest_path": "v0.2.0/metadata/runtime-manifest.json",
             "immutable_revision": t4["wheel"]["hf_revision"], "native_sha256": t4["wheel"]["sha256"],
             "publication_status": "ACCEPTED", "acceptance_status": "historical exact artifact"},
            {"sdk_version": "0.3.1", "gpu_profile": "L4x4/SM89", "vllm": "0.30.0", "python_abi": "cp312",
             "cuda": "12.9", "torch": "2.13.0+cu129", "manifest_path": None, "immutable_revision": None,
             "native_sha256": l4_lock["native"]["sha256"], "publication_status": "PUBLISHED_UNVERIFIED",
             "acceptance_status": "historical candidate evidence does not cover exact published SDK"},
            {"sdk_version": "0.3.2", "gpu_profile": "L4x4/SM89", "vllm": "0.30.0", "python_abi": "cp312",
             "cuda": "12.9", "torch": "2.13.0+cu129", "manifest_path": "v0.3.2/l4x4/metadata/runtime-manifest.json",
             "immutable_revision": None, "native_sha256": l4_lock["native"]["sha256"],
             "wheelhouse_delivery": "publisher-index", "publication_status": "PREPARING", "acceptance_status": "NOT RUN"},
        ],
    }
    write_json(STAGING / "releases.json", releases)
    (STAGING / "LICENSE").write_bytes((ROOT / "LICENSE").read_bytes())
    (STAGING / "NOTICE").write_text(
        "kaggle-vllm SDK is Apache-2.0. Native vLLM and each dependency wheel retain their own licenses.\n"
        "This staging directory is not authorization to redistribute NVIDIA or other third-party binaries.\n"
        "See reports/v032-production-release/13-license-provenance-audit.md before publication.\n"
    )
    (STAGING / "README.md").write_text(
        "# kaggle-vllm binary release namespaces\n\n"
        "Historical root paths and pinned commits remain valid. v0.2.0 is T4x2/cu128; "
        "v0.3.2/l4x4 is the proposed cu129 native wheel and metadata namespace. "
        "Its 168 locked dependencies are fetched from the original PyPI, PyTorch and FlashInfer publishers "
        "using exact checksums; dependency wheels are not mirrored here. "
        "The v0.3.2 namespace is not public or accepted until all sources, notices, "
        "immutable revision and manual Kaggle evidence are verified.\n"
    )
    rows = []
    for record in l4_lock["wheels"]:
        path = WHEELHOUSE / record["group"] / record["filename"]
        if not path.is_file():
            rows.append({"filename": record["filename"], "status": "MISSING"})
            continue
        with zipfile.ZipFile(path) as archive:
            metadata_path = next((x for x in archive.namelist() if x.endswith(".dist-info/METADATA")), None)
            metadata = email.message_from_bytes(archive.read(metadata_path)) if metadata_path else {}
            licenses = [x for x in archive.namelist() if "license" in x.lower() or "notice" in x.lower()]
        expression = metadata.get("License-Expression") or metadata.get("License") or ""
        rows.append({"filename": record["filename"], "distribution": record["distribution"],
                     "version": record["version"], "license_metadata": expression[:160],
                     "license_file_count": len(licenses), "license_files": licenses[:8],
                     "redistribution_status": "REVIEW_REQUIRED"})
    write_json(REPORT / "license-wheel-inventory.json", rows)
    print(f"Prepared metadata for {len(l4_lock['wheels'])} dependency wheels; no binary copied or uploaded")


if __name__ == "__main__":
    main()
