#!/usr/bin/env python3
"""Generate retained V4 and current post-Phi M4-BATCH-2 source freezes."""

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
IMPLEMENTATION_COMMIT = "fafac6ffc8fac6745d67732fcceb128cad1f5f5a"
NOTEBOOK_PIN_COMMIT = "f6b547bc182c5febc2bcfe9600e3c09098ab0b94"
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
        "schema_version": "kaggle-vllm-m4-batch-source-freeze-v4",
        "status": "FROZEN_AFTER_MINISTRAL_R00_FOR_M4_BATCH_2_CONTINUATION",
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
            "path": "research/M4_BATCH_SOURCE_FREEZE_V3.json",
            "sha256": sha256_file(
                repository / "research/M4_BATCH_SOURCE_FREEZE_V3.json"
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
                "status": "M4_BATCH_2_POST_MINISTRAL_CONTINUATION_READY",
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
            "12-canonical/1-resource-gated/47-not-executed reconciliation and "
            "duplicate-prevention guard for subsequent M4-BATCH-2 execution."
        ),
    }


def build_freeze_v5(repository: Path = ROOT) -> dict:
    """Build the post-Phi freeze without duplicating the freeze generator."""

    implementation_commit = "e50bb5b7d75856528528345d4755026fa548b194"
    notebook_pin_commit = "845957d31e4475207e79b9b4f74dd5188ba71079"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v5",
            "status": "FROZEN_AFTER_PHI_R01_FOR_M4_BATCH_2_CONTINUATION",
            "reconciliation_date_utc": "2026-09-13",
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V4.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V4.json"
                ),
                "status": "RETAINED_UNCHANGED",
            },
            "batch_notebook_sha256": hashlib.sha256(notebook_blob).hexdigest(),
            "batch_notebook_source_digest": source_digest,
            "batch_runner_sha256": _blob_sha256(
                repository, implementation_commit, "scripts/kaggle_m4_execute_batch.py"
            ),
            "base_shard_runner_sha256": _blob_sha256(
                repository,
                implementation_commit,
                "scripts/kaggle_m4_multimodel_crossover.py",
            ),
            "m4_execution_plan_sha256": _blob_sha256(
                repository, implementation_commit, "research/M4_EXECUTION_PLAN.json"
            ),
            "model_matrix_sha256": _blob_sha256(
                repository, implementation_commit, "research/model_matrix.json"
            ),
            "protocol_sha256": _blob_sha256(
                repository, implementation_commit, "research/m4_protocol.json"
            ),
            "batch_plan_sha256": _blob_sha256(
                repository, implementation_commit, BATCH_PLAN
            ),
            "principal_queue_sha256": _blob_sha256(
                repository, implementation_commit, PRINCIPAL_QUEUE
            ),
            "protocol_amendment_sha256": _blob_sha256(
                repository, implementation_commit, AMENDMENT
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "15-canonical/1-resource-gated/44-not-executed reconciliation and "
                "duplicate-prevention guard for subsequent M4-BATCH-2 execution."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_2_POST_PHI_R01_CONTINUATION_READY"
    )
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/M4_BATCH_SOURCE_FREEZE_V5.json",
    )
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(build_freeze_v5(), indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
