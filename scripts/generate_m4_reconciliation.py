#!/usr/bin/env python3
"""Generate the authoritative M4 no-rerun ledger and remaining schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.generate_m4_batch_plan_v2 import build_continuation_plan
    from scripts.generate_m4_queue import build_queue
else:
    from generate_m4_batch_plan_v2 import build_continuation_plan
    from generate_m4_queue import build_queue


STATE_MAP = {
    "PRINCIPAL_SHARD_PRESERVED": "CANONICAL_PRESERVED",
    "FAILED_RESOURCE_GATE": "FAILED_RESOURCE_GATE",
    "FAILED_OTHER_REVIEW_REQUIRED": "FAILED_OTHER_REVIEW_REQUIRED",
    "VERIFIED_LOCAL_STAGING_PENDING_PROMOTION": "VERIFIED_LOCAL_STAGING_PENDING_PROMOTION",
    "QUEUED": "NOT_EXECUTED",
}
TERMINAL_STATES = {
    "CANONICAL_PRESERVED",
    "FAILED_RESOURCE_GATE",
    "FAILED_OTHER_REVIEW_REQUIRED",
}


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def build_reconciliation(
    execution_plan: dict[str, Any],
    evidence: dict[str, Any],
    historical_batch_plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    queue = build_queue(execution_plan, evidence)
    remaining_plan = build_continuation_plan(historical_batch_plan, queue)
    active = [row for row in queue["queue"] if row["active_order"] is not None]
    if len(active) != 60 or sum(row["serving_cells"] for row in active) != 720:
        raise ValueError("M4 reconciliation requires the unchanged 60-shard/720-cell design")

    rows = []
    for row in active:
        try:
            state = STATE_MAP[row["status"]]
        except KeyError as error:
            raise ValueError(
                f"unsupported reconciliation state: {row['shard_id']}={row['status']}"
            ) from error
        evidence_record = evidence.get("principal_shards", {}).get(row["shard_id"])
        rows.append(
            {
                "active_order": row["active_order"],
                "shard_id": row["shard_id"],
                "model_key": row["model_key"],
                "workload": row["workload"],
                "repetition": row["repetition"],
                "serving_cells": row["serving_cells"],
                "state": state,
                "terminal_for_normal_continuation": state in TERMINAL_STATES,
                "evidence_zip_sha256": (
                    evidence_record.get("evidence_zip_sha256")
                    if evidence_record
                    else None
                ),
                "batch_id": evidence_record.get("batch_id") if evidence_record else None,
                "session_id": (
                    evidence_record.get("kaggle_session_id")
                    if evidence_record
                    else None
                ),
            }
        )

    by_state = {
        state: [row["shard_id"] for row in rows if row["state"] == state]
        for state in STATE_MAP.values()
    }
    counts = {state: len(ids) for state, ids in by_state.items()}
    if sum(counts.values()) != 60:
        raise ValueError("M4 reconciliation does not total 60 logical shards")
    if len({row["shard_id"] for row in rows}) != 60:
        raise ValueError("M4 reconciliation contains duplicate logical shards")

    batches = []
    for batch in remaining_plan["batches"]:
        batches.append(
            {
                "batch_id": batch["batch_id"],
                "repetition": batch["repetition"],
                "model_key": batch["model_order"][0],
                "ordered_shard_ids": batch["ordered_shard_ids"],
                "logical_shard_count": batch["logical_shard_count"],
                "serving_cell_count": batch["serving_cell_count"],
                "already_settled_exclusions": [
                    *batch["already_completed_skips"],
                    *batch["review_required_exclusions"],
                ],
            }
        )
    scheduled = [shard for batch in batches for shard in batch["ordered_shard_ids"]]
    if len(scheduled) != len(set(scheduled)):
        raise ValueError("remaining schedule duplicates logical shards")
    if set(scheduled) != set(by_state["NOT_EXECUTED"]):
        raise ValueError("remaining schedule does not exactly cover not-executed shards")

    reconciliation = {
        "schema_version": "kaggle-vllm-m4-execution-reconciliation-v1",
        "status": "AUTHORITATIVE_TRACKED_RECONCILIATION",
        "authority_order": [
            "CRYPTOGRAPHICALLY_VERIFIED_PRESERVED_EVIDENCE",
            "VERIFIED_LOCAL_INGESTION_STAGING",
            "TRACKED_MACHINE_READABLE_RESEARCH_STATUS",
            "PR_PROSE",
            "FILENAMES_OR_UI_SCREENSHOTS",
        ],
        "source_execution_plan": "research/M4_EXECUTION_PLAN.json",
        "source_evidence_status": "research/M4_EVIDENCE_STATUS.json",
        "source_principal_queue": "research/M4_PRINCIPAL_EXECUTION_QUEUE.json",
        "logical_design": {"logical_shards": 60, "serving_cells": 720},
        "counts": counts,
        "counts_total": sum(counts.values()),
        "shards_by_state": by_state,
        "shards": rows,
        "no_rerun_shards": [
            row["shard_id"] for row in rows if row["terminal_for_normal_continuation"]
        ],
        "verified_local_staging_pending_promotion": by_state[
            "VERIFIED_LOCAL_STAGING_PENDING_PROMOTION"
        ],
        "duplicate_shard_ids": [],
        "conflicting_terminal_records": [],
        "remaining_batch_count": len(batches),
        "remaining_batches": batches,
        "next_batch_id": batches[0]["batch_id"] if batches else None,
    }
    return reconciliation, remaining_plan


def markdown(reconciliation: dict[str, Any]) -> str:
    counts = reconciliation["counts"]
    lines = [
        "# M4 execution reconciliation",
        "",
        "Status: **AUTHORITATIVE TRACKED NO-RERUN LEDGER**.",
        "",
        "Evidence takes precedence in this order: cryptographically verified preserved",
        "evidence, verified local staging, tracked machine-readable status, PR prose,",
        "then filenames or UI screenshots.",
        "",
        "## Exact 60-shard state",
        "",
        f"- Canonical preserved: {counts['CANONICAL_PRESERVED']}",
        f"- Failed resource gate: {counts['FAILED_RESOURCE_GATE']}",
        f"- Failed other/review-required: {counts['FAILED_OTHER_REVIEW_REQUIRED']}",
        (
            "- Verified local staging pending promotion: "
            f"{counts['VERIFIED_LOCAL_STAGING_PENDING_PROMOTION']}"
        ),
        f"- Not executed: {counts['NOT_EXECUTED']}",
        f"- Total: {reconciliation['counts_total']} logical shards / 720 serving cells",
        "",
        "Every canonical or reviewed terminal shard is excluded from normal continuation.",
        "A resource-gated shard is settled negative evidence, not a zero-throughput result.",
        "",
        "## No-rerun ledger",
        "",
    ]
    lines.extend(f"- `{shard_id}`" for shard_id in reconciliation["no_rerun_shards"])
    lines.extend(
        [
            "",
            "## Deterministic remaining batches",
            "",
            "| Order | Batch ID | Rep | Model | Shards | Cells | Settled exclusions |",
            "|---:|---|---:|---|---:|---:|---|",
        ]
    )
    for index, batch in enumerate(reconciliation["remaining_batches"], 1):
        exclusions = ", ".join(
            f"`{item}`" for item in batch["already_settled_exclusions"]
        ) or "—"
        lines.append(
            f"| {index} | `{batch['batch_id']}` | {batch['repetition']} | "
            f"`{batch['model_key']}` | {batch['logical_shard_count']} | "
            f"{batch['serving_cell_count']} | {exclusions} |"
        )
    lines.extend(
        [
            "",
            f"Next: `M4_BATCH_ID={reconciliation['next_batch_id']}`.",
            "",
            "The detailed shard IDs and order are machine-readable in",
            "`M4_REMAINING_EXECUTION_PLAN.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-plan", type=Path, default=Path("research/M4_EXECUTION_PLAN.json")
    )
    parser.add_argument(
        "--evidence", type=Path, default=Path("research/M4_EVIDENCE_STATUS.json")
    )
    parser.add_argument(
        "--historical-batch-plan",
        type=Path,
        default=Path("research/M4_BATCH_EXECUTION_PLAN.json"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("research/M4_EXECUTION_RECONCILIATION.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("research/M4_EXECUTION_RECONCILIATION.md"),
    )
    parser.add_argument(
        "--remaining-plan-output",
        type=Path,
        default=Path("research/M4_REMAINING_EXECUTION_PLAN.json"),
    )
    args = parser.parse_args()
    reconciliation, remaining_plan = build_reconciliation(
        load_object(args.execution_plan),
        load_object(args.evidence),
        load_object(args.historical_batch_plan),
    )
    args.json_output.write_text(
        json.dumps(reconciliation, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(markdown(reconciliation), encoding="utf-8")
    args.remaining_plan_output.write_text(
        json.dumps(remaining_plan, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
