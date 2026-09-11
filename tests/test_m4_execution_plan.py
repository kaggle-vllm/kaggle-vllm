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


def test_gemma_canonical_negative_remains_auditable_and_blocks_principal():
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
    assert {
        key: status["compatibility"][key]["evidence_zip_sha256"]
        for key in sorted(accepted)
    } == {
        "llama32_3b": "37288c24065ddd2383c5e8dfebf08b61ca589bb2b8a37e9ae253ef9e6a1f9db4",
        "ministral3_3b_bf16": "516e37c160ca98a49795870aeeeeb8a381cd8780924f4d9d8836305ad68c77cc",
        "phi4_mini": "7a20d7058cdcf7362704acbc5513db65eb865b864fd47b555f72465a32a0ebfe",
        "qwen25_3b": "cd3c45dddf19830649b03a931cee0247d6f8b4ebbc953c433e3ade563b271ecb",
    }

    gemma = status["compatibility"]["gemma3_4b"]
    assert gemma["status"] == "UNSUPPORTED_DTYPE_INTERSECTION_ON_SM75_FROZEN_STACK"
    assert gemma["canonical"] is True
    assert gemma["principal_eligible"] is False
    assert gemma["throughput"] is None
    assert gemma["principal_shard_status"] == "SKIPPED_BY_COMPATIBILITY_GATE"
    assert status["milestone_status"] == "IN_PROGRESS"
    assert status["principal_matrix_status"] == "IN_PROGRESS"
    assert status["principal_preserved_shards"] == 3
    assert status["principal_remaining_shards"] == 57
    assert status["principal_review_required_shards"] == 1
    assert status["next_principal_shard"] == "phi4_mini-short-r00"
    assert status["principal_shards"]["qwen25_3b-short-r00"]["canonical"] is True
    assert (
        status["principal_shards"]["qwen25_3b-short-r00"][
            "paper_aggregate_eligible"
        ]
        is False
    )

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
    assert review["canonical"] is True
    assert review["next_action"]["decision"] == "NO_MORE_GEMMA_EXECUTION"
