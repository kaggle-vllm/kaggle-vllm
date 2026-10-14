"""Assemble the small Kaggle upload directory without copying CUDA payloads."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
NAME = "kaggle-vllm-v030-production-acceptance"
SDK = DIST / "kaggle_vllm-0.3.0-py3-none-any.whl"
SDK_SHA = "08ff056319dcfbf1a020272e447fae39964d347065550c5d415ebd10cb89f5e3"
RUNTIME_SHA = "0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73"
PURE_DIR = "kaggle-vllm-v030-pinned-pure-runtime-wheels"
HOUSE = ROOT.parent / "kaggle-vllm-v030-wheelhouse-cp312-cu129-v2" / "build-deps"
PURE_WHEELS = {
    "jinja2-3.1.6-py3-none-any.whl": "85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67",
    "packaging-26.3-py3-none-any.whl": "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c",
}
FILES = {
    SDK: SDK.name,
    ROOT / "kaggle-notebooks/kaggle_vllm_0_3_0_l4x4_final_acceptance.ipynb": "kaggle_vllm_0_3_0_l4x4_final_acceptance.ipynb",
    ROOT / "scripts/production_acceptance_v030.py": "production_acceptance_v030.py",
    ROOT / "scripts/production_target_probe_v030.py": "production_target_probe_v030.py",
    ROOT / "scripts/fresh_kaggle_gemma_v030.py": "fresh_kaggle_gemma_v030.py",
    ROOT / "release/README-v0.3.0-production-acceptance.md": "README-RUN-ME-FIRST.md",
    ROOT / "release/v0.3.0-acceptance-evidence-schema.json": "EVIDENCE_SCHEMA.json",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if digest(SDK) != SDK_SHA:
        raise RuntimeError("SDK wheel identity changed; rebuild acceptance package identities first")
    target = DIST / NAME
    with tempfile.TemporaryDirectory(prefix=NAME + ".tmp-", dir=DIST) as tmp:
        stage = Path(tmp) / NAME
        stage.mkdir()
        for source, name in FILES.items():
            if not source.is_file():
                raise RuntimeError(f"missing acceptance source: {source}")
            shutil.copy2(source, stage / name)
        pure = stage / PURE_DIR
        pure.mkdir()
        for filename, expected in PURE_WHEELS.items():
            source = HOUSE / filename
            if not source.is_file() or digest(source) != expected:
                raise RuntimeError(f"pinned small pure runtime wheel identity mismatch: {source}")
            shutil.copy2(source, pure / filename)
        (stage / "SDK_WHEEL_SHA256SUMS.txt").write_text(f"{SDK_SHA}  {SDK.name}\n")
        (stage / "RUNTIME_V3_SHA256SUMS.txt").write_text(f"{RUNTIME_SHA}  kaggle-vllm-v030-l4x4-runtime-v3.tar.gz\n")
        provenance = {"schema_version": 3, "status": "PREPARED_NOT_EXECUTED", "default_runtime_mode": "SPLIT_V2_KAGGLE_INPUTS", "sdk_sha256": SDK_SHA, "runtime_v3_sha256": RUNTIME_SHA, "native_wheel_sha256": "9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8", "runtime_archive_in_package": False, "pure_runtime_wheels": PURE_WHEELS, "fa2_model_runtime_pass": False, "fresh_kaggle_core_release_pass": False}
        (stage / "PACKAGE_STATUS.json").write_text(json.dumps(provenance, indent=2) + "\n")
        lines = [f"{digest(path)}  {path.relative_to(stage)}" for path in sorted(stage.rglob("*")) if path.is_file()]
        (stage / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
        if any(path.suffix in (".so", ".safetensors") or path.name.endswith(("runtime-v3.tar.gz", "vllm-0.30.0-cp312-cp312-linux_x86_64.whl")) for path in stage.rglob("*")):
            raise RuntimeError("large/native artifact entered small acceptance package")
        if target.exists():
            if not target.is_dir() or target.is_symlink() or not (target / "PACKAGE_STATUS.json").is_file():
                raise RuntimeError(f"refusing unexpected existing acceptance path: {target}")
            backup = DIST / (NAME + ".superseded-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            if backup.exists():
                raise RuntimeError(f"refusing to replace existing backup: {backup}")
            os.replace(target, backup)
            try:
                os.replace(stage, target)
            except OSError:
                os.replace(backup, target)
                raise
        else:
            os.replace(stage, target)
    print(json.dumps({"directory": str(target), "file_count": sum(p.is_file() for p in target.rglob("*")), "bytes": sum(p.stat().st_size for p in target.rglob("*") if p.is_file())}))


if __name__ == "__main__":
    main()
