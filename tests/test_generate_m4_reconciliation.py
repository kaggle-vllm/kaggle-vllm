from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_m4_reconciliation import build_reconciliation

ROOT = Path(__file__).resolve().parents[1]


def test_reconciliation_is_exact_and_deterministic() -> None:
    inputs = (
        json.loads((ROOT / "research/M4_EXECUTION_PLAN.json").read_text()),
        json.loads((ROOT / "research/M4_EVIDENCE_STATUS.json").read_text()),
        json.loads((ROOT / "research/M4_BATCH_EXECUTION_PLAN.json").read_text()),
    )
    first, first_plan = build_reconciliation(*inputs)
    second, second_plan = build_reconciliation(*inputs)
    assert first == second
    assert first_plan == second_plan
    assert first["counts"] == {
        "CANONICAL_PRESERVED": 9,
        "FAILED_RESOURCE_GATE": 1,
        "FAILED_OTHER_REVIEW_REQUIRED": 0,
        "VERIFIED_LOCAL_STAGING_PENDING_PROMOTION": 0,
        "NOT_EXECUTED": 50,
    }
    assert first["counts_total"] == 60
    assert len(first["shards"]) == len({row["shard_id"] for row in first["shards"]}) == 60
    assert first["remaining_batch_count"] == 17
    assert first["next_batch_id"] == "fill-r00-ministral"


def test_reconciliation_excludes_every_settled_shard_from_schedule() -> None:
    reconciliation = json.loads(
        (ROOT / "research/M4_EXECUTION_RECONCILIATION.json").read_text()
    )
    scheduled = {
        shard_id
        for batch in reconciliation["remaining_batches"]
        for shard_id in batch["ordered_shard_ids"]
    }
    assert scheduled.isdisjoint(reconciliation["no_rerun_shards"])
    assert len(scheduled) == 50
    assert "qwen25_3b-prefill_heavy-r00" not in scheduled
    assert "llama32_3b-short-r00" not in scheduled
