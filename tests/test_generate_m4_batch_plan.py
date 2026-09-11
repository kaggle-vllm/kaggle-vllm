import json
from copy import deepcopy
from pathlib import Path

from kaggle_vllm.research.m4_batch import verify_batch_source_freeze
from scripts.generate_m4_batch_plan import build_batch_plan
from scripts.generate_m4_batch_source_freeze_v2 import build_freeze as build_freeze_v2

ROOT = Path(__file__).resolve().parents[1]


def _plan() -> dict:
    queue = json.loads(
        (ROOT / "research/M4_PRINCIPAL_EXECUTION_QUEUE.json").read_text()
    )
    historical = deepcopy(queue)
    by_id = {row["shard_id"]: row for row in historical["queue"]}
    by_id["qwen25_3b-balanced-r00"]["status"] = "QUEUED"
    by_id["qwen25_3b-prefill_heavy-r00"]["status"] = "QUEUED"
    historical.update(
        {
            "preserved_shards": 2,
            "preserved_serving_cells": 24,
            "queued_shards": 58,
            "queued_serving_cells": 696,
        }
    )
    return build_batch_plan(historical)


def test_batch_plan_preserves_matrix_and_canonical_skips() -> None:
    plan = _plan()
    assert plan["scientific_matrix"]["logical_shards"] == 60
    assert plan["scientific_matrix"]["serving_cells"] == 720
    assert plan["remaining_logical_shards"] == 58
    assert plan["batch_ids"] == ["fill-r00", "fill-r01", "r02", "r03", "r04"]
    assert [batch["logical_shard_count"] for batch in plan["batches"]] == [
        11,
        11,
        12,
        12,
        12,
    ]
    assert plan["batches"][0]["already_completed_skips"] == [
        "qwen25_3b-short-r00"
    ]
    assert plan["batches"][1]["already_completed_skips"] == [
        "qwen25_3b-short-r01"
    ]


def test_batch_order_is_deterministic_counterbalanced_and_repetition_blocked() -> None:
    first = _plan()
    second = _plan()
    assert first == second
    model_orders = [batch["model_order"] for batch in first["batches"]]
    assert model_orders[:4] == [
        ["qwen25_3b", "phi4_mini", "llama32_3b", "ministral3_3b_bf16"],
        ["phi4_mini", "llama32_3b", "ministral3_3b_bf16", "qwen25_3b"],
        ["llama32_3b", "ministral3_3b_bf16", "qwen25_3b", "phi4_mini"],
        ["ministral3_3b_bf16", "qwen25_3b", "phi4_mini", "llama32_3b"],
    ]
    seen: dict[tuple[str, str], set[int]] = {}
    for batch in first["batches"]:
        repetitions = {item["repetition"] for item in batch["execution_order"]}
        assert repetitions <= {batch["repetition"]}
        identities = [
            (item["model_key"], item["workload"])
            for item in batch["execution_order"]
        ]
        assert len(identities) == len(set(identities))
        for identity in identities:
            seen.setdefault(identity, set()).add(batch["repetition"])
    assert seen[("qwen25_3b", "short")] == {2, 3, 4}
    assert all(
        len(repetitions) == 5
        for identity, repetitions in seen.items()
        if identity != ("qwen25_3b", "short")
    )


def test_batch_plan_uses_per_gpu_binary_vram_guard() -> None:
    resource = _plan()["resource_policy"]
    assert resource["hard_vram_limit_mib_per_gpu"] == 14848
    assert resource["independent_per_gpu_enforcement"] is True
    assert resource["disk_safety_reserve_bytes"] == 2 * 1024**3
    assert resource["base_shard_runner_projected_evidence_allowance_bytes"] == (
        3_000_000_000
    )
    assert resource["maximum_batch_wall_clock_seconds"] == int(10.5 * 3600)


def test_batch_source_freeze_generation_is_deterministic() -> None:
    expected = json.loads((ROOT / "research/M4_BATCH_SOURCE_FREEZE.json").read_text())
    assert verify_batch_source_freeze(ROOT) == expected


def test_batch_v2_source_freeze_generation_is_deterministic() -> None:
    expected = json.loads(
        (ROOT / "research/M4_BATCH_SOURCE_FREEZE_V2.json").read_text()
    )
    assert build_freeze_v2(ROOT) == expected
    assert build_freeze_v2(ROOT) == build_freeze_v2(ROOT)
