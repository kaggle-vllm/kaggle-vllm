from __future__ import annotations

from scripts.generate_m4_execution_plan import MODEL_ORDER, build_plan


def test_plan_has_ordered_compatibility_principal_and_crosscheck_shards():
    plan = build_plan({"models": {key: {} for key in MODEL_ORDER}})
    assert [item["model_key"] for item in plan["compatibility_order"]] == list(MODEL_ORDER)
    assert len(plan["principal_order"]) == 5 * 3 * 5
    assert len({item["shard_id"] for item in plan["principal_order"]}) == 75
    assert len(plan["guidellm_crosscheck_order"]) == 3 * 5 * 2
    assert plan["refinement_rule"]["resulting_total_repetitions"] == 10
