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
