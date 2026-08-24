import json

import pytest

from kaggle_vllm.exceptions import ShardedModelError
from kaggle_vllm.sharding import inspect_sharded_model


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
    assert inspection.rank_count == 2
    assert len(inspection.shards) == 4
    assert inspection.total_size == 8
    assert "rank-specific files" in inspection.warnings[0]


def test_inspect_sharded_model_rejects_missing_shards(tmp_path):
    with pytest.raises(ShardedModelError, match="no rank-specific"):
        inspect_sharded_model(tmp_path)
