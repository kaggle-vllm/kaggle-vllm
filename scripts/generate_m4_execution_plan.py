#!/usr/bin/env python3
"""Generate the deterministic M4 and GuideLLM Kaggle shard order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ORDER = (
    "qwen25_3b",
    "phi4_mini",
    "llama32_3b",
    "ministral3_3b_bf16",
    "gemma3_4b",
)
WORKLOAD_ORDER = ("short", "balanced", "prefill_heavy")


def build_plan(model_matrix: dict) -> dict:
    models = model_matrix["models"]
    if set(MODEL_ORDER) - set(models):
        raise ValueError("model matrix is missing an M4 candidate")
    compatibility = [
        {
            "order": index + 1,
            "shard_id": f"compat-{key}",
            "model_key": key,
            "mode": "compatibility",
            "workload": "short",
            "repetition": 0,
            "prerequisite": "MODEL_ACCESS_AND_LICENSE_ACCEPTANCE_IF_GATED",
        }
        for index, key in enumerate(MODEL_ORDER)
    ]
    principal = []
    order = 1
    for key in MODEL_ORDER:
        for workload in WORKLOAD_ORDER:
            for repetition in range(5):
                principal.append(
                    {
                        "order": order,
                        "shard_id": f"{key}-{workload}-r{repetition:02d}",
                        "model_key": key,
                        "mode": "principal",
                        "workload": workload,
                        "repetition": repetition,
                        "prerequisite": f"compat-{key}:COMPATIBILITY_PASS",
                    }
                )
                order += 1
    guidellm = []
    order = 1
    for concurrency in (1, 16, 64):
        for repetition in range(5):
            for tp in (1, 2):
                guidellm.append(
                    {
                        "order": order,
                        "shard_id": f"qwen25_3b-balanced-c{concurrency:02d}-r{repetition:02d}-tp{tp}",
                        "model_key": "qwen25_3b",
                        "workload": "balanced",
                        "concurrency": concurrency,
                        "repetition": repetition,
                        "tensor_parallel_size": tp,
                        "prerequisite": "accepted Qwen M4 principal cells at this configuration",
                    }
                )
                order += 1
    return {
        "schema_version": "kaggle-vllm-m4-execution-plan-v1",
        "status": "PREPARED_FOR_KAGGLE",
        "session_policy": "one shard in one fresh T4 x2 Kaggle session",
        "compatibility_order": compatibility,
        "principal_order": principal,
        "guidellm_crosscheck_order": guidellm,
        "refinement_rule": {
            "status": "DATA_DEPENDENT_NOT_YET_SCHEDULED",
            "candidate_points": [12, 20, 24, 48],
            "selection": "Only points bracketing an observed sign change or capacity boundary",
            "repetitions": [5, 6, 7, 8, 9],
            "resulting_total_repetitions": 10,
        },
        "exclusion_rule": "Do not run principal shards for a model whose compatibility gate failed; preserve that negative evidence.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    model_matrix = json.loads(args.model_matrix.read_text(encoding="utf-8"))
    plan = build_plan(model_matrix)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
