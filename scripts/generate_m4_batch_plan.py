#!/usr/bin/env python3
"""Derive deterministic repetition-blocked M4 batch orchestration plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _rotate(values: list[str], offset: int) -> list[str]:
    offset %= len(values)
    return values[offset:] + values[:offset]


def build_batch_plan(queue: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in queue["queue"] if row["active_order"] is not None]
    models = list(queue["principal_models"])
    workloads = list(queue["workloads"])
    if len(rows) != 60 or queue["active_serving_cells"] != 720:
        raise ValueError("batching requires the unchanged 60-shard/720-cell matrix")
    if len(models) != 4 or len(workloads) != 3:
        raise ValueError("unexpected active M4 model/workload population")

    batches = []
    for repetition in range(queue["repetitions_per_model_workload"]):
        batch_id = f"fill-r{repetition:02d}" if repetition < 2 else f"r{repetition:02d}"
        model_order = _rotate(models, repetition)
        workload_order = _rotate(workloads, repetition)
        rank = {
            (model, workload): (model_order.index(model), workload_order.index(workload))
            for model in models
            for workload in workloads
        }
        repetition_rows = [row for row in rows if row["repetition"] == repetition]
        repetition_rows.sort(key=lambda row: rank[(row["model_key"], row["workload"])])
        preserved = [
            row["shard_id"]
            for row in repetition_rows
            if row["status"] == "PRINCIPAL_SHARD_PRESERVED"
        ]
        pending = [
            row for row in repetition_rows if row["status"] != "PRINCIPAL_SHARD_PRESERVED"
        ]
        if any(row["status"] != "QUEUED" for row in pending):
            raise ValueError(f"unexpected active shard status in {batch_id}")
        identities = [(row["model_key"], row["workload"]) for row in repetition_rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"duplicate model/workload identity in {batch_id}")
        batches.append(
            {
                "batch_id": batch_id,
                "label": f"BATCH_{batch_id.replace('-', '_').upper()}",
                "repetition": repetition,
                "model_order": model_order,
                "workload_order_within_model": workload_order,
                "all_logical_shard_ids": [row["shard_id"] for row in repetition_rows],
                "ordered_shard_ids": [row["shard_id"] for row in pending],
                "execution_order": [
                    {
                        "within_session_order": index,
                        "shard_id": row["shard_id"],
                        "model_key": row["model_key"],
                        "workload": row["workload"],
                        "repetition": row["repetition"],
                        "serving_cells": row["serving_cells"],
                    }
                    for index, row in enumerate(pending, 1)
                ],
                "already_completed_skips": preserved,
                "logical_shard_count": len(pending),
                "serving_cell_count": sum(row["serving_cells"] for row in pending),
            }
        )

    remaining = sum(batch["logical_shard_count"] for batch in batches)
    if remaining != queue["queued_shards"]:
        raise ValueError("batch rows do not exactly cover the active queue")
    return {
        "schema_version": "kaggle-vllm-m4-batch-execution-plan-v1",
        "status": "FROZEN_BEFORE_BULK_PRINCIPAL_COLLECTION",
        "protocol_amendment": {
            "version": "M4-BATCH-1",
            "path": "research/M4_BATCH_PROTOCOL_AMENDMENT.md",
            "reason": "The frozen one-session-per-shard rule is amended transparently for operational orchestration only.",
        },
        "source_queue": "research/M4_PRINCIPAL_EXECUTION_QUEUE.json",
        "scientific_matrix": {
            "models": models,
            "workloads": workloads,
            "repetitions": list(range(queue["repetitions_per_model_workload"])),
            "concurrency": queue["concurrency"],
            "tensor_parallel_sizes": queue["tensor_parallel_sizes"],
            "logical_shards": 60,
            "serving_cells": 720,
            "fresh_server_per_cell": True,
        },
        "ordering_strategy": {
            "name": "pre_frozen_cyclic_model_and_workload_rotation",
            "performance_adaptive": False,
            "model_rotation_offset": "repetition modulo active model count",
            "workload_rotation_offset": "repetition modulo workload count",
            "model_grouping_reason": "reuse one ordinary pinned Hugging Face checkpoint across that model's workloads",
        },
        "session_blocking_policy": {
            "one_repetition_per_batch": True,
            "same_model_workload_repetitions_in_one_session": False,
            "kaggle_allocation_is_blocking_variable": True,
            "required_metadata": [
                "batch_id",
                "session_id",
                "repetition",
                "within_session_order",
                "start_utc",
                "end_utc",
                "gpu_names",
                "gpu_uuids",
                "runtime_identity",
            ],
        },
        "resource_policy": {
            "hard_vram_limit_mib_per_gpu": 14848,
            "independent_per_gpu_enforcement": True,
            "disk_capacity_source": "shutil.disk_usage(/kaggle/working)",
            "disk_safety_reserve_bytes": 2 * 1024**3,
            "projected_evidence_bytes_per_shard": 16 * 1024**2,
            "base_shard_runner_projected_evidence_allowance_bytes": 3_000_000_000,
            "pre_shard_free_bytes_rule": "max(selected weights + base runner 3 GB allowance, uncached selected weights + compact evidence + 2 GiB reserve)",
            "maximum_batch_wall_clock_seconds": 37800,
            "minimum_remaining_seconds_to_start_shard": 5400,
            "idle_gpu_memory_limit_mib_per_gpu": 64,
        },
        "failure_policy": {
            "unexpected_or_scientific_failure": "preserve completed/current evidence, finalize manifest and outer ZIP, then stop",
            "wall_clock_or_disk_guard": "graceful stop; remaining rows are NOT_EXECUTED_IN_THIS_BATCH_ATTEMPT",
            "unexpected_residual_gpu_process": "do not kill unrelated processes; preserve evidence and stop",
            "partial_completion_is_not_failure": True,
            "overwrite_existing_evidence": False,
        },
        "cache_policy": {
            "model_source": "ordinary pinned Hugging Face checkpoints from research/model_matrix.json",
            "controlled_hf_home": "/kaggle/working/hf-cache",
            "group_workloads_by_model": True,
            "delete_only_exact_completed_model_repo_cache": True,
            "delete_weights_before_evidence_verified_and_zipped": False,
            "historical_qwen_tp2_sharded_state_allowed": False,
            "historical_dataset_repo_allowed_as_model_source": False,
            "forbidden_model_source": "waqasm86/kaggle-vllm-models",
        },
        "resume_policy": {
            "automatic_cross_session_resume": False,
            "prior_manifest_import_required": True,
            "continuation_batch_id_format": "<batch_id>-part<N>",
            "new_physical_allocation_requires_new_session_id": True,
        },
        "batch_ids": [batch["batch_id"] for batch in batches],
        "remaining_logical_shards": remaining,
        "batches": batches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("research/M4_PRINCIPAL_EXECUTION_QUEUE.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/M4_BATCH_EXECUTION_PLAN.json"),
    )
    args = parser.parse_args()
    plan = build_batch_plan(load_object(args.queue))
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
