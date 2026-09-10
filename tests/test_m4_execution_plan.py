from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.generate_m4_execution_plan import MODEL_ORDER, build_plan


def test_plan_has_ordered_compatibility_principal_and_crosscheck_shards():
    plan = build_plan({"models": {key: {} for key in MODEL_ORDER}})
    assert [item["model_key"] for item in plan["compatibility_order"]] == list(MODEL_ORDER)
    assert len(plan["principal_order"]) == 5 * 3 * 5
    assert len({item["shard_id"] for item in plan["principal_order"]}) == 75
    assert len(plan["guidellm_crosscheck_order"]) == 3 * 5 * 2
    assert plan["refinement_rule"]["resulting_total_repetitions"] == 10


def test_m4_source_freeze_hashes_match_repository_files():
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads((root / "research/M4_SOURCE_FREEZE.json").read_text())
    for relative, expected in freeze["notebooks"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    for field, relative in (
        ("execution_plan_sha256", "research/M4_EXECUTION_PLAN.json"),
        ("model_matrix_sha256", "research/model_matrix.json"),
        ("protocol_sha256", "research/m4_protocol.json"),
    ):
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == freeze[field]


def test_gemma_diagnostic_negative_remains_auditable_and_blocks_principal():
    root = Path(__file__).resolve().parents[1]
    status = json.loads((root / "research/M4_EVIDENCE_STATUS.json").read_text())
    review = json.loads(
        (root / "research/M4_GEMMA3_DIAGNOSTIC_REVIEW.json").read_text()
    )
    plan = json.loads((root / "research/M4_EXECUTION_PLAN.json").read_text())

    accepted = {"qwen25_3b", "phi4_mini", "ministral3_3b_bf16", "llama32_3b"}
    assert {
        key for key, value in status["compatibility"].items()
        if value["status"] == "COMPATIBILITY_PASS"
    } == accepted

    gemma = status["compatibility"]["gemma3_4b"]
    assert gemma["status"] == "REPRODUCED_NEGATIVE_DIAGNOSTIC_EVIDENCE"
    assert gemma["canonical"] is False
    assert gemma["principal_eligible"] is False
    assert status["milestone_status"] == "IN_PROGRESS"
    assert status["principal_matrix_status"] == "NOT_STARTED"

    planned_gemma = [
        shard for shard in plan["principal_order"]
        if shard["model_key"] == "gemma3_4b"
    ]
    assert len(planned_gemma) == 15
    assert all(
        shard["prerequisite"] == "compat-gemma3_4b:COMPATIBILITY_PASS"
        for shard in planned_gemma
    )

    observations = review["reproduced_observations"]
    assert observations["measured_requests_issued"] == 0
    assert observations["throughput"] is None
    assert observations["oom_observed"] is False
    assert review["canonical"] is False
    assert review["next_action"]["decision"] == (
        "ONE_FINAL_CLEAN_GEMMA_CANONICALIZATION_RUN_REQUIRED"
    )
