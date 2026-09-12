#!/usr/bin/env python3
"""Generate the M4-BATCH-2 reconciled no-rerun source freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from kaggle_vllm.research.m4_batch import notebook_source_digest
from kaggle_vllm.research.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_COMMIT = "1189cd0ed71e3f12de9a80a0a03278af0888a130"
NOTEBOOK_PIN_COMMIT = "0f35e5388cd537ffa1fdaabeee20e260acac9e0b"
BATCH_PLAN = "research/M4_BATCH_EXECUTION_PLAN_V2.json"
PRINCIPAL_QUEUE = "research/M4_PRINCIPAL_EXECUTION_QUEUE.json"
AMENDMENT = "research/M4_BATCH_PROTOCOL_AMENDMENT_V2.md"


def _blob(repository: Path, commit: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=repository
    )


def _blob_sha256(repository: Path, commit: str, relative: str) -> str:
    return hashlib.sha256(_blob(repository, commit, relative)).hexdigest()


def build_freeze(repository: Path = ROOT) -> dict:
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, NOTEBOOK_PIN_COMMIT, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))
    return {
        "schema_version": "kaggle-vllm-m4-batch-source-freeze-v3",
        "status": "FROZEN_FOR_RECONCILED_M4_BATCH_2_CONTINUATION",
        "protocol_amendment_version": "M4-BATCH-2",
        "protocol_amendment_date_utc": "2026-09-11",
        "reconciliation_date_utc": "2026-09-12",
        "protocol_amendment_path": AMENDMENT,
        "batch_plan_path": BATCH_PLAN,
        "principal_queue_path": PRINCIPAL_QUEUE,
        "package_version": "0.2.0",
        "implementation_source_commit": IMPLEMENTATION_COMMIT,
        "notebook_pin_commit": NOTEBOOK_PIN_COMMIT,
        "historical_batch_2_freeze": {
            "path": "research/M4_BATCH_SOURCE_FREEZE_V2.json",
            "sha256": sha256_file(
                repository / "research/M4_BATCH_SOURCE_FREEZE_V2.json"
            ),
            "status": "RETAINED_UNCHANGED",
        },
        "execution_paths": {
            "single_shard": {
                "status": "UNCHANGED_AND_SUPPORTED",
                "source_freeze": "research/M4_SOURCE_FREEZE.json",
                "source_freeze_sha256": sha256_file(
                    repository / "research/M4_SOURCE_FREEZE.json"
                ),
            },
            "batch_orchestrated": {
                "status": "M4_BATCH_2_RECONCILED_CONTINUATION_READY",
                "notebook": notebook_relative,
                "runner": "scripts/kaggle_m4_execute_batch.py",
                "duplicate_prevention": "FAIL_CLOSED_AGAINST_FROZEN_PRINCIPAL_QUEUE",
            },
        },
        "batch_notebook_sha256": hashlib.sha256(notebook_blob).hexdigest(),
        "batch_notebook_source_digest": source_digest,
        "batch_runner_sha256": _blob_sha256(
            repository, IMPLEMENTATION_COMMIT, "scripts/kaggle_m4_execute_batch.py"
        ),
        "base_shard_runner_sha256": _blob_sha256(
            repository,
            IMPLEMENTATION_COMMIT,
            "scripts/kaggle_m4_multimodel_crossover.py",
        ),
        "m4_execution_plan_sha256": _blob_sha256(
            repository, IMPLEMENTATION_COMMIT, "research/M4_EXECUTION_PLAN.json"
        ),
        "model_matrix_sha256": _blob_sha256(
            repository, IMPLEMENTATION_COMMIT, "research/model_matrix.json"
        ),
        "protocol_sha256": _blob_sha256(
            repository, IMPLEMENTATION_COMMIT, "research/m4_protocol.json"
        ),
        "batch_plan_sha256": _blob_sha256(
            repository, IMPLEMENTATION_COMMIT, BATCH_PLAN
        ),
        "principal_queue_sha256": _blob_sha256(
            repository, IMPLEMENTATION_COMMIT, PRINCIPAL_QUEUE
        ),
        "protocol_amendment_sha256": _blob_sha256(
            repository, IMPLEMENTATION_COMMIT, AMENDMENT
        ),
        "settled_shard_policy": {
            "canonical_preserved": "REFUSE_EXECUTION",
            "failed_resource_gate": "REFUSE_EXECUTION",
            "failed_other_review_required": "REFUSE_EXECUTION",
            "zero_remaining_shards": "REFUSE_EXECUTION",
            "pr_prose_authoritative": False,
        },
        "model_source_policy": {
            "source": "ordinary pinned Hugging Face checkpoints in research/model_matrix.json",
            "historical_qwen_tp2_sharded_state_allowed": False,
            "waqasm86_kaggle_vllm_models_allowed": False,
        },
        "evidence_boundary": (
            "This freeze contains no new GPU measurements. It binds the reviewed "
            "9-canonical/1-resource-gated/50-not-executed reconciliation and "
            "duplicate-prevention guard for subsequent M4-BATCH-2 execution."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/M4_BATCH_SOURCE_FREEZE_V3.json",
    )
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_freeze(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
