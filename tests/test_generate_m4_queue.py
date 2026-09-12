import json
from pathlib import Path

from scripts.generate_m4_queue import build_queue

ROOT = Path(__file__).resolve().parents[1]


def test_queue_preserves_negative_compatibility_and_historical_shards() -> None:
    plan = json.loads((ROOT / "research/M4_EXECUTION_PLAN.json").read_text())
    evidence = json.loads((ROOT / "research/M4_EVIDENCE_STATUS.json").read_text())
    queue = build_queue(plan, evidence)

    assert queue["compatibility_population"] == 5
    assert queue["principal_model_count"] == 4
    assert queue["active_shards"] == 60
    assert queue["active_serving_cells"] == 720
    assert queue["preserved_shards"] == 9
    assert queue["preserved_serving_cells"] == 108
    assert queue["queued_shards"] == 50
    assert queue["queued_serving_cells"] == 600
    assert queue["review_required_shards"] == 1
    assert queue["remaining_shards"] == 51
    assert queue["skipped_gemma_shards"] == 15
    assert queue["next_shard_id"] == "ministral3_3b_bf16-short-r00"
    assert queue["queue"][0]["status"] == "PRINCIPAL_SHARD_PRESERVED"
    assert queue["queue"][1]["status"] == "PRINCIPAL_SHARD_PRESERVED"
    by_id = {row["shard_id"]: row for row in queue["queue"]}
    assert by_id["qwen25_3b-balanced-r00"]["status"] == "PRINCIPAL_SHARD_PRESERVED"
    assert by_id["qwen25_3b-prefill_heavy-r00"]["status"] == "FAILED_RESOURCE_GATE"
    assert by_id["phi4_mini-short-r00"]["status"] == "PRINCIPAL_SHARD_PRESERVED"
    assert by_id["phi4_mini-balanced-r00"]["status"] == "PRINCIPAL_SHARD_PRESERVED"
    assert (
        by_id["phi4_mini-prefill_heavy-r00"]["status"]
        == "PRINCIPAL_SHARD_PRESERVED"
    )
    assert by_id["llama32_3b-short-r00"]["status"] == "PRINCIPAL_SHARD_PRESERVED"
    assert by_id["llama32_3b-balanced-r00"]["status"] == "PRINCIPAL_SHARD_PRESERVED"
    assert (
        by_id["llama32_3b-prefill_heavy-r00"]["status"]
        == "PRINCIPAL_SHARD_PRESERVED"
    )
    assert queue["queue"][0]["expected_artifact"] == (
        "qwen25_3b-short-r00-principal.zip"
    )
    assert "source-equivalence" in queue["queue"][0]["executed_notebook_role"]
    assert "runtime-identity" in queue["queue"][0]["runtime_json_role"]
    gemma = [row for row in queue["queue"] if row["model_key"] == "gemma3_4b"]
    assert len(gemma) == 15
    assert {row["status"] for row in gemma} == {"SKIPPED_BY_COMPATIBILITY_GATE"}
    assert all(row["active_order"] is None for row in gemma)


def test_negative_compatibility_is_not_a_zero_throughput_result() -> None:
    evidence = json.loads((ROOT / "research/M4_EVIDENCE_STATUS.json").read_text())
    gemma = evidence["compatibility"]["gemma3_4b"]
    assert gemma["status"] == "UNSUPPORTED_DTYPE_INTERSECTION_ON_SM75_FROZEN_STACK"
    assert gemma["throughput"] is None
    assert gemma["principal_eligible"] is False
    assert evidence["milestone_status"] == "IN_PROGRESS"
