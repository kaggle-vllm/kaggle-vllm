#!/usr/bin/env python3
"""Generate the deterministic M4 batch source-freeze record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_vllm.research.m4_batch import notebook_source_digest
from kaggle_vllm.research.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_COMMIT = "8cd6b224684f03810116c9a91370f25a7c2be297"
NOTEBOOK_PIN_COMMIT = "5e0415261418801b1c45735af3a8056dee330be3"


def build_freeze(repository: Path = ROOT) -> dict:
    notebook = repository / "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    return {
        "schema_version": "kaggle-vllm-m4-batch-source-freeze-v1",
        "status": "FROZEN_FOR_KAGGLE_NO_BATCH_GPU_RESULTS",
        "protocol_amendment_version": "M4-BATCH-1",
        "protocol_amendment_date_utc": "2026-09-11",
        "package_version": "0.2.0",
        "implementation_source_commit": IMPLEMENTATION_COMMIT,
        "notebook_pin_commit": NOTEBOOK_PIN_COMMIT,
        "execution_paths": {
            "single_shard": {
                "status": "UNCHANGED_AND_SUPPORTED",
                "source_freeze": "research/M4_SOURCE_FREEZE.json",
                "source_freeze_sha256": sha256_file(
                    repository / "research/M4_SOURCE_FREEZE.json"
                ),
                "notebook_sha256": sha256_file(
                    repository / "kaggle-notebooks/kaggle_vllm_m4_execute_shard.ipynb"
                ),
            },
            "batch_orchestrated": {
                "status": "FROZEN_BEFORE_BULK_PRINCIPAL_COLLECTION",
                "notebook": "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb",
                "runner": "scripts/kaggle_m4_execute_batch.py",
            },
        },
        "batch_notebook_sha256": sha256_file(notebook),
        "batch_notebook_source_digest": notebook_source_digest(notebook),
        "batch_runner_sha256": sha256_file(
            repository / "scripts/kaggle_m4_execute_batch.py"
        ),
        "base_shard_runner_sha256": sha256_file(
            repository / "scripts/kaggle_m4_multimodel_crossover.py"
        ),
        "m4_execution_plan_sha256": sha256_file(
            repository / "research/M4_EXECUTION_PLAN.json"
        ),
        "model_matrix_sha256": sha256_file(repository / "research/model_matrix.json"),
        "protocol_sha256": sha256_file(repository / "research/m4_protocol.json"),
        "batch_plan_sha256": sha256_file(
            repository / "research/M4_BATCH_EXECUTION_PLAN.json"
        ),
        "protocol_amendment_sha256": sha256_file(
            repository / "research/M4_BATCH_PROTOCOL_AMENDMENT.md"
        ),
        "model_source_policy": {
            "source": "ordinary pinned Hugging Face checkpoints in research/model_matrix.json",
            "historical_qwen_tp2_sharded_state_allowed": False,
            "waqasm86_kaggle_vllm_models_allowed": False,
        },
        "evidence_boundary": (
            "This file freezes an unexecuted orchestration path. It contains no GPU "
            "measurements and does not promote any future shard without independent "
            "batch and inner-shard ingestion."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/M4_BATCH_SOURCE_FREEZE.json",
    )
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_freeze(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
