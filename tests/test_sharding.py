import json

import pytest

from kaggle_vllm.exceptions import ShardedModelError
from kaggle_vllm.sharding import copy_model_metadata, inspect_sharded_model


def test_inspect_sharded_model_reports_ranks_parts_and_stale_hf_index(tmp_path):
    for rank in (0, 1):
        for part in (0, 1):
            (tmp_path / f"model-rank-{rank}-part-{part}.safetensors").write_bytes(
                bytes([rank, part])
            )
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}}),
        encoding="utf-8",
    )
    inspection = inspect_sharded_model(tmp_path)
    assert inspection.valid
    assert inspection.topology_errors == ()
    assert inspection.rank_count == 2
    assert len(inspection.shards) == 4
    assert inspection.total_size == 8
    assert "rank-specific files" in inspection.warnings[0]


def test_inspect_sharded_model_rejects_missing_shards(tmp_path):
    with pytest.raises(ShardedModelError, match="no rank-specific"):
        inspect_sharded_model(tmp_path)


def test_inspect_rejects_mismatched_tp_topology(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-rank-0-part-0.safetensors").write_bytes(b"rank0")
    inspection = inspect_sharded_model(tmp_path, expected_tensor_parallel_size=2)
    assert not inspection.valid
    assert "checkpoint has 1 ranks" in inspection.topology_errors[0]


def test_inspect_rejects_symlinked_members(tmp_path):
    outside = tmp_path.parent / "outside.safetensors"
    outside.write_bytes(b"outside")
    (tmp_path / "model-rank-0-part-0.safetensors").symlink_to(outside)
    with pytest.raises(ShardedModelError, match="refusing symlink"):
        inspect_sharded_model(tmp_path)


def test_copy_metadata_rejects_nested_symlink(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    nested = source / "metadata"
    nested.mkdir(parents=True)
    destination.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    (nested / "link.txt").symlink_to(outside)
    with pytest.raises(ShardedModelError, match="nested symlink"):
        copy_model_metadata(source, destination)
