"""Check rank completeness and metadata without loading large tensor payloads."""
import importlib.util
import json
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/gemma_tp4_sharded_state_kaggle.py"
spec = importlib.util.spec_from_file_location("gemma_tp4_trial", SCRIPT)
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def _shard(path: Path, key: str, value: float) -> None:
    header = json.dumps({key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + struct.pack("<f", value))


def test_inspect_shards_requires_four_ranks_and_records_tensor_metadata(tmp_path):
    for rank in range(4):
        _shard(tmp_path / f"model-rank-{rank}-part-0.safetensors", f"weight_{rank}", float(rank))
    manifest = trial.inspect_shards(tmp_path)
    assert manifest["ranks"] == [0, 1, 2, 3]
    assert manifest["tensor_keys_by_rank"] == {str(rank): 1 for rank in range(4)}
    assert manifest["shards"][2]["tensors"]["weight_2"] == {"shape": [1], "dtype": "F32"}
    assert all(len(item["sha256"]) == 64 for item in manifest["shards"])
    (tmp_path / "model-rank-3-part-0.safetensors").unlink()
    with pytest.raises(RuntimeError, match="rank 3 missing"):
        trial.inspect_shards(tmp_path)
