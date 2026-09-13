from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.assemble_m4_evidence import verify_shard


def create_shard(path: Path):
    path.mkdir()
    raw = path / "m4-raw.json"
    raw.write_text(json.dumps({"schema_version": "kaggle-vllm-m4-serving-raw-v1", "rows": []}), encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    (path / "SHA256SUMS.txt").write_text(f"{digest}  m4-raw.json\n", encoding="utf-8")


def test_verify_shard_checks_every_manifest_entry(tmp_path: Path):
    shard = tmp_path / "shard"
    create_shard(shard)
    verified = verify_shard(shard)
    assert verified["payload"]["rows"] == []
    (shard / "m4-raw.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Exception, match="checksum mismatch"):
        verify_shard(shard)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nested/file"])
def test_verify_shard_rejects_nonflat_manifest_names(tmp_path: Path, name: str):
    shard = tmp_path / "shard"
    create_shard(shard)
    (shard / "SHA256SUMS.txt").write_text(f"{'0' * 64}  {name}\n", encoding="utf-8")
    with pytest.raises(Exception, match="malformed"):
        verify_shard(shard)
