#!/usr/bin/env python3
"""Derive the active M4 queue without changing the frozen execution plan."""

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


def build_queue(plan: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    compatibility = evidence["compatibility"]
    rows = []
    active_order = 0
    for historical in plan["principal_order"]:
        model_status = compatibility[historical["model_key"]]
        eligible = model_status.get("status") == "COMPATIBILITY_PASS"
        if eligible:
            active_order += 1
        rows.append(
            {
                "historical_order": historical["order"],
                "active_order": active_order if eligible else None,
                "shard_id": historical["shard_id"],
                "model_key": historical["model_key"],
                "workload": historical["workload"],
                "repetition": historical["repetition"],
                "concurrency": [1, 4, 8, 16, 32, 64],
                "tensor_parallel_sizes": [1, 2],
                "serving_cells": 12,
                "expected_artifact": f"{historical['shard_id']}-principal.zip",
                "executed_notebook_role": (
                    "downloaded executed copy of the frozen M4 notebook; "
                    "required for source-equivalence audit"
                ),
                "runtime_json_role": (
                    "separately downloaded bootstrap runtime.json; required for "
                    "runtime-identity audit"
                ),
                "status": "QUEUED" if eligible else "SKIPPED_BY_COMPATIBILITY_GATE",
                "skip_reason": None if eligible else "FAILED_FROZEN_COMPATIBILITY_GATE",
            }
        )
    queued = [row for row in rows if row["status"] == "QUEUED"]
    skipped = [row for row in rows if row["status"] != "QUEUED"]
    return {
        "schema_version": "kaggle-vllm-m4-principal-queue-v1",
        "source_plan": "research/M4_EXECUTION_PLAN.json",
        "source_plan_unchanged": True,
        "compatibility_population": 5,
        "principal_models": list(dict.fromkeys(row["model_key"] for row in queued)),
        "principal_model_count": 4,
        "workloads": ["short", "balanced", "prefill_heavy"],
        "repetitions_per_model_workload": 5,
        "concurrency": [1, 4, 8, 16, 32, 64],
        "tensor_parallel_sizes": [1, 2],
        "queued_shards": len(queued),
        "queued_serving_cells": sum(row["serving_cells"] for row in queued),
        "skipped_gemma_shards": len(skipped),
        "next_shard_id": queued[0]["shard_id"] if queued else None,
        "session_policy": "one shard in one fresh Kaggle T4 x2 session",
        "planning_estimate": {
            "wall_clock_hours_per_shard_range": [0.5, 1.5],
            "dual_gpu_hours_total_range": [60, 180],
            "basis": "12 fresh-server cells per shard; observed compatibility startup plus workload-dependent request time",
            "not_a_measured_result": True,
        },
        "protocol_reduction_recommendation": "KEEP_FROZEN_60_SHARD_DESIGN",
        "queue": rows,
    }


def markdown(queue: dict[str, Any]) -> str:
    lines = [
        "# M4 principal execution queue",
        "",
        "Compatibility is closed. Use the unchanged frozen notebook in one fresh",
        "Kaggle T4 x2 session per row and change only the `M4_SHARD_ID` secret value.",
        "For every active row, also download the executed copy of",
        "`kaggle_vllm_m4_execute_shard.ipynb` and the separately generated",
        "`/kaggle/working/kaggle-vllm-runtime/runtime.json`; both are mandatory",
        "inputs to the local provenance audit.",
        "",
        f"Next: `M4_SHARD_ID={queue['next_shard_id']}`",
        "",
        "| Active | M4_SHARD_ID | Model | Workload | Rep | Status | Artifact |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for row in queue["queue"]:
        active = row["active_order"] if row["active_order"] is not None else "—"
        lines.append(
            f"| {active} | `{row['shard_id']}` | `{row['model_key']}` | "
            f"{row['workload']} | {row['repetition']} | {row['status']} | "
            f"`{row['expected_artifact']}` |"
        )
    lines.extend(
        [
            "",
            "Every queued shard contains concurrency 1, 4, 8, 16, 32, and 64 for",
            "TP1 and TP2 (12 serving cells). The 15 Gemma rows preserve frozen",
            "historical intent and are not executable. Their performance is N/A.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("research/M4_EXECUTION_PLAN.json"))
    parser.add_argument("--evidence", type=Path, default=Path("research/M4_EVIDENCE_STATUS.json"))
    parser.add_argument("--json-output", type=Path, default=Path("research/M4_PRINCIPAL_EXECUTION_QUEUE.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("research/M4_PRINCIPAL_EXECUTION_QUEUE.md"))
    args = parser.parse_args()
    queue = build_queue(load_object(args.plan), load_object(args.evidence))
    args.json_output.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(queue), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
