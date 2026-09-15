from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.assemble_m4_evidence import assemble, verify_shard

ROOT = Path(__file__).resolve().parents[1]


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


def test_assembly_accepts_multiple_immutable_scientifically_identical_sources(
    tmp_path: Path,
) -> None:
    commits = [
        "42bf096c032e2c6be1e2fa3d573c7c86ac589ba2",
        "8cd6b224684f03810116c9a91370f25a7c2be297",
    ]
    shards = []
    for index, commit in enumerate(commits):
        shard = tmp_path / f"shard-{index}"
        shard.mkdir()
        raw = shard / "m4-raw.json"
        raw.write_text(
            json.dumps(
                {
                    "schema_version": "kaggle-vllm-m4-serving-raw-v1",
                    "status": "executed",
                    "mode": "principal",
                    "source": {"commit": commit, "dirty": False},
                    "rows": [],
                }
            ),
            encoding="utf-8",
        )
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        (shard / "SHA256SUMS.txt").write_text(
            f"{digest}  m4-raw.json\n", encoding="utf-8"
        )
        shards.append(shard)
    combined, _ = assemble(shards, repository=ROOT)
    assert combined["source_commits"] == sorted(commits)
    assert len(combined["source_freezes"]) == 2
    assert combined["canonical_shards"] == 2
    assert combined["resource_boundary_shards"] == 0
