from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_m4_batch_plan_v2 import build_continuation_plan

ROOT = Path(__file__).resolve().parents[1]


def test_continuations_are_model_scoped_repetition_blocked_and_complete() -> None:
    historical = json.loads(
        (ROOT / "research/M4_BATCH_EXECUTION_PLAN.json").read_text()
    )
    queue = json.loads(
        (ROOT / "research/M4_PRINCIPAL_EXECUTION_QUEUE.json").read_text()
    )
    plan = build_continuation_plan(historical, queue)
    assert plan["scientific_matrix"]["logical_shards"] == 60
    assert plan["scientific_matrix"]["serving_cells"] == 720
    assert plan["preserved_logical_shards"] == 6
    assert plan["remaining_logical_shards"] == 54
    assert plan["continuation_executable_shards"] == 53
    assert plan["review_required_shard_ids"] == [
        "qwen25_3b-prefill_heavy-r00"
    ]
    assert len(plan["batches"]) == 18
    assert plan["batch_ids"][0] == "fill-r00-llama"

    shard_ids = []
    for batch in plan["batches"]:
        assert len(batch["model_order"]) == 1
        assert {row["repetition"] for row in batch["execution_order"]} == {
            batch["repetition"]
        }
        assert {row["model_key"] for row in batch["execution_order"]} == set(
            batch["model_order"]
        )
        assert [row["within_session_order"] for row in batch["execution_order"]] == list(
            range(1, len(batch["execution_order"]) + 1)
        )
        shard_ids.extend(batch["ordered_shard_ids"])
    assert len(shard_ids) == len(set(shard_ids)) == 53
    assert "qwen25_3b-balanced-r00" not in shard_ids
    assert "qwen25_3b-prefill_heavy-r00" not in shard_ids
    assert "phi4_mini-short-r00" not in shard_ids
    assert "phi4_mini-balanced-r00" not in shard_ids
    assert "phi4_mini-prefill_heavy-r00" not in shard_ids


def test_first_continuation_is_exact_llama_r00_group() -> None:
    plan = build_continuation_plan(
        json.loads((ROOT / "research/M4_BATCH_EXECUTION_PLAN.json").read_text()),
        json.loads((ROOT / "research/M4_PRINCIPAL_EXECUTION_QUEUE.json").read_text()),
    )
    first = plan["batches"][0]
    assert first["batch_id"] == "fill-r00-llama"
    assert first["repetition"] == 0
    assert first["already_completed_skips"] == []
    assert first["review_required_exclusions"] == []
    assert first["ordered_shard_ids"] == [
        "llama32_3b-short-r00",
        "llama32_3b-balanced-r00",
        "llama32_3b-prefill_heavy-r00",
    ]
    assert first["serving_cell_count"] == 36
