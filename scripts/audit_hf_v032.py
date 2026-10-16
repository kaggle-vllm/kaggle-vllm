"""Read-only Hugging Face binary inventory for the 0.3.2 release review."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

REPO = "waqasm86/kaggle-vllm-binaries"
API = f"https://huggingface.co/api/models/{REPO}"
OUTPUT = Path(__file__).resolve().parents[1] / "reports/v032-production-release/04-huggingface-existing-inventory.json"


def read_json(url: str):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def main() -> None:
    info = read_json(API)
    revision = info["sha"]
    rows = read_json(f"{API}/tree/{revision}?recursive=true")
    result = []
    for row in rows:
        if row["type"] != "file":
            continue
        path = row["path"]
        lfs = row.get("lfs") or {}
        digest = lfs.get("oid")
        verification = "Hub LFS/Xet object identity"
        if not digest:
            url = f"https://huggingface.co/{REPO}/resolve/{revision}/{quote(path, safe='/')}"
            with urlopen(url, timeout=30) as response:
                raw = response.read()
            digest = hashlib.sha256(raw).hexdigest()
            verification = "downloaded bytes SHA256"
        family = "0.2.0" if path.startswith(("vllm-0.18", "kaggle-t4x2", "vllm-wheel")) else "0.3.0/0.3.1" if path.startswith("v0.3.0/") else "historical/metadata"
        result.append({
            "path": path, "size": row["size"], "sha256": digest,
            "sha256_verification": verification, "git_oid": row.get("oid"),
            "xet_hash": row.get("xetHash"), "version": family,
            "intended_sdk_release": family, "native_abi": "cp312" if path.startswith("vllm-") or "/native/" in path else None,
            "cuda": "12.8" if family == "0.2.0" else "12.9" if family == "0.3.0/0.3.1" else None,
            "gpu_architecture": "SM75" if family == "0.2.0" else "SM89" if family == "0.3.0/0.3.1" else None,
            "relationship": "native vLLM wheel" if path.endswith(".whl") and "vllm-" in path else "SDK or repository metadata",
            "license": "third-party binary redistribution review pending" if path.endswith(".whl") and "vllm-" in path else "inspect package metadata",
            "current_consumers": ["kaggle-vllm==0.2.0 pinned HF revision"] if path.startswith("vllm-0.18") else ["kaggle-vllm 0.3.1 historical path"] if path.startswith("v0.3.0/") else [],
            "safe_to_move": False,
            "safe_to_copy_additively": bool(lfs),
        })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"repository": REPO, "revision": revision, "file_count": len(result), "files": result}, indent=2) + "\n")
    print(f"{len(result)} files at {revision} -> {OUTPUT}")


if __name__ == "__main__":
    main()
