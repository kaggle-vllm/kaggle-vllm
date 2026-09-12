#!/usr/bin/env python3
"""Generate deterministic model-scoped M4 continuation batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_BATCH_SUFFIX = {
    "qwen25_3b": "qwen",
    "phi4_mini": "phi",
    "llama32_3b": "llama",
    "ministral3_3b_bf16": "ministral",
}


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def build_continuation_plan(
    historical_plan: dict[str, Any], queue: dict[str, Any]
) -> dict[str, Any]:
    if historical_plan.get("schema_version") != "kaggle-vllm-m4-batch-execution-plan-v1":
        raise ValueError("continuation generation requires the historical M4-BATCH-1 plan")
    if queue.get("active_shards") != 60 or queue.get("active_serving_cells") != 720:
        raise ValueError("continuation requires the unchanged 60-shard/720-cell matrix")
    rows = {
        row["shard_id"]: row
        for row in queue["queue"]
        if row["active_order"] is not None
    }
    allowed = {"QUEUED", "PRINCIPAL_SHARD_PRESERVED", "FAILED_RESOURCE_GATE"}
    if any(row["status"] not in allowed for row in rows.values()):
        raise ValueError("continuation queue contains an unsupported active status")

    batches = []
    covered: set[str] = set()
    for historical_batch in historical_plan["batches"]:
        repetition = historical_batch["repetition"]
        for model_key in historical_batch["model_order"]:
            model_rows = [
                rows[shard_id]
                for shard_id in historical_batch["all_logical_shard_ids"]
                if rows[shard_id]["model_key"] == model_key
            ]
            pending = [row for row in model_rows if row["status"] == "QUEUED"]
            if not pending:
                continue
            preserved = [
                row["shard_id"]
                for row in model_rows
                if row["status"] == "PRINCIPAL_SHARD_PRESERVED"
            ]
            review_required = [
                row["shard_id"]
                for row in model_rows
                if row["status"] == "FAILED_RESOURCE_GATE"
            ]
            batch_id = (
                f"{historical_batch['batch_id']}-{MODEL_BATCH_SUFFIX[model_key]}"
            )
            execution_order = [
                {
                    "within_session_order": index,
                    "shard_id": row["shard_id"],
                    "model_key": row["model_key"],
                    "workload": row["workload"],
                    "repetition": row["repetition"],
                    "serving_cells": row["serving_cells"],
                }
                for index, row in enumerate(pending, 1)
            ]
            batch = {
                "batch_id": batch_id,
                "label": f"CONTINUATION_{batch_id.replace('-', '_').upper()}",
                "protocol_amendment_version": "M4-BATCH-2",
                "parent_batch_id": historical_batch["batch_id"],
                "continuation_group": MODEL_BATCH_SUFFIX[model_key],
                "repetition": repetition,
                "model_order": [model_key],
                "workload_order_within_model": [row["workload"] for row in model_rows],
                "all_logical_shard_ids": [row["shard_id"] for row in model_rows],
                "ordered_shard_ids": [row["shard_id"] for row in pending],
                "execution_order": execution_order,
                "already_completed_skips": preserved,
                "review_required_exclusions": review_required,
                "logical_shard_count": len(pending),
                "serving_cell_count": sum(row["serving_cells"] for row in pending),
            }
            batches.append(batch)
            overlap = covered.intersection(batch["ordered_shard_ids"])
            if overlap:
                raise ValueError(f"continuation duplicates logical shards: {sorted(overlap)}")
            covered.update(batch["ordered_shard_ids"])

    queued = {row["shard_id"] for row in rows.values() if row["status"] == "QUEUED"}
    if covered != queued:
        raise ValueError("continuation groups do not exactly cover queued logical shards")
    review_required = sorted(
        row["shard_id"]
        for row in rows.values()
        if row["status"] == "FAILED_RESOURCE_GATE"
    )
    return {
        "schema_version": "kaggle-vllm-m4-batch-execution-plan-v2",
        "status": "DERIVED_FROM_CURRENT_PRINCIPAL_QUEUE",
        "protocol_amendment": {
            "version": "M4-BATCH-2",
            "path": "research/M4_BATCH_PROTOCOL_AMENDMENT_V2.md",
            "supersedes_for_future_execution": "M4-BATCH-1",
            "historical_plan_retained": "research/M4_BATCH_EXECUTION_PLAN.json",
            "reason": "Model-scoped continuation isolates failures while preserving valid prior evidence.",
        },
        "source_queue": "research/M4_PRINCIPAL_EXECUTION_QUEUE.json",
        "scientific_matrix": historical_plan["scientific_matrix"],
        "session_blocking_policy": historical_plan["session_blocking_policy"],
        "resource_policy": historical_plan["resource_policy"],
        "cache_policy": historical_plan["cache_policy"],
        "failure_policy": {
            **historical_plan["failure_policy"],
            "blast_radius": "one model/repetition continuation group",
            "failed_review_required_shards_are_not_automatic_skips": True,
        },
        "continuation_policy": {
            "grouping": "one model within one repetition per physical Kaggle session",
            "performance_adaptive": False,
            "observed_failure_informed_operational_grouping": True,
            "logical_shard_ids_unchanged": True,
            "canonical_shards_skipped": True,
            "review_required_shards_excluded_not_promoted": True,
            "new_physical_allocation_requires_new_session_id": True,
        },
        "preserved_logical_shards": queue["preserved_shards"],
        "remaining_logical_shards": queue["remaining_shards"],
        "continuation_executable_shards": queue["queued_shards"],
        "review_required_shard_ids": review_required,
        "batch_ids": [batch["batch_id"] for batch in batches],
        "batches": batches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-plan",
        type=Path,
        default=Path("research/M4_BATCH_EXECUTION_PLAN.json"),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("research/M4_PRINCIPAL_EXECUTION_QUEUE.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/M4_BATCH_EXECUTION_PLAN_V2.json"),
    )
    args = parser.parse_args()
    plan = build_continuation_plan(
        load_object(args.historical_plan), load_object(args.queue)
    )
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
