#!/usr/bin/env python3
"""Generate retained V4-V18 and current M4-BATCH-3 source freezes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from kaggle_vllm.research.m4_batch import (
    BATCH_NOTEBOOK_SOURCE_FILES,
    notebook_source_digest,
    validate_batch_notebook_commit_consistency,
)
from kaggle_vllm.research.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_COMMIT = "fafac6ffc8fac6745d67732fcceb128cad1f5f5a"
NOTEBOOK_PIN_COMMIT = "f6b547bc182c5febc2bcfe9600e3c09098ab0b94"
BATCH_PLAN = "research/M4_BATCH_EXECUTION_PLAN_V2.json"
PRINCIPAL_QUEUE = "research/M4_PRINCIPAL_EXECUTION_QUEUE.json"
AMENDMENT = "research/M4_BATCH_PROTOCOL_AMENDMENT_V2.md"
NOTEBOOK = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"


def _blob(repository: Path, commit: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=repository
    )


def _blob_sha256(repository: Path, commit: str, relative: str) -> str:
    return hashlib.sha256(_blob(repository, commit, relative)).hexdigest()


def render_batch_notebook(
    repository: Path, implementation_commit: str, notebook_path: Path | None = None
) -> dict:
    """Render one clean notebook from the exact implementation commit snapshot."""

    notebook_path = notebook_path or repository / NOTEBOOK
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    bootstrap = [cell for cell in notebook["cells"] if cell.get("id") == "bootstrap"]
    if len(bootstrap) != 1:
        raise ValueError("batch notebook must have one bootstrap cell")
    source = bootstrap[0]["source"]
    commit_lines = [
        index
        for index, line in enumerate(source)
        if line.startswith("EXPECTED_SOURCE_COMMIT = ")
    ]
    digest_lines = [
        index
        for index, line in enumerate(source)
        if line.startswith("BATCH_NOTEBOOK_SOURCE_DIGEST = ")
    ]
    manifest_starts = [
        index for index, line in enumerate(source) if line == "EXPECTED_FILES = {\n"
    ]
    if len(commit_lines) != 1 or len(digest_lines) != 1 or len(manifest_starts) != 1:
        raise ValueError("batch notebook source identity assignments are not unique")
    manifest_start = manifest_starts[0]
    try:
        manifest_end = source.index("}\n", manifest_start + 1)
    except ValueError as exc:
        raise ValueError("batch notebook EXPECTED_FILES mapping is not closed") from exc

    expected_files = {
        relative: _blob_sha256(repository, implementation_commit, relative)
        for relative in BATCH_NOTEBOOK_SOURCE_FILES
    }
    source[commit_lines[0]] = (
        f"EXPECTED_SOURCE_COMMIT = '{implementation_commit}'\n"
    )
    source[digest_lines[0]] = "BATCH_NOTEBOOK_SOURCE_DIGEST = '" + "0" * 64 + "'\n"
    source[manifest_start : manifest_end + 1] = [
        "EXPECTED_FILES = {\n",
        *[
            f"    {relative!r}: {expected_files[relative]!r},\n"
            for relative in BATCH_NOTEBOOK_SOURCE_FILES
        ],
        "}\n",
    ]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb") as temporary:
        temporary.write(json.dumps(notebook, indent=1) + "\n")
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))
    digest_lines = [
        index
        for index, line in enumerate(source)
        if line.startswith("BATCH_NOTEBOOK_SOURCE_DIGEST = ")
    ]
    source[digest_lines[0]] = f"BATCH_NOTEBOOK_SOURCE_DIGEST = '{source_digest}'\n"
    notebook_path.write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8"
    )
    identity = validate_batch_notebook_commit_consistency(repository, notebook_path)
    return {
        **identity,
        "batch_notebook_sha256": sha256_file(notebook_path),
        "batch_notebook_source_digest": source_digest,
    }


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


def build_freeze_v6(repository: Path = ROOT) -> dict:
    """Build the post-Llama freeze through the existing versioned generator."""

    implementation_commit = "e8119613cc98c990a2097cd97fce8af51de78220"
    notebook_pin_commit = "8d5ae0df5b2baec5b6e3c88eaf73a73bcab32d77"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v5(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v6",
            "status": "FROZEN_AFTER_LLAMA_R01_FOR_M4_BATCH_2_CONTINUATION",
            "reconciliation_date_utc": "2026-09-14",
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V5.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V5.json"
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
                "18-canonical/1-resource-gated/41-not-executed reconciliation and "
                "duplicate-prevention guard for subsequent M4-BATCH-2 execution."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_2_POST_LLAMA_R01_CONTINUATION_READY"
    )
    return freeze


def build_freeze_v7(repository: Path = ROOT) -> dict:
    """Build the post-Ministral-r01 freeze through the versioned generator."""

    implementation_commit = "c05ca0db682074f29db9459d0cd9d50e162b34e6"
    notebook_pin_commit = "b79961a7f4ff4e54ebf03ce917dafe12f8f191e5"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v6(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v7",
            "status": "FROZEN_AFTER_MINISTRAL_R01_FOR_M4_BATCH_2_CONTINUATION",
            "reconciliation_date_utc": "2026-09-14",
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V6.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V6.json"
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
                "21-canonical/1-resource-gated/38-not-executed reconciliation and "
                "duplicate-prevention guard for subsequent M4-BATCH-2 execution."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_2_POST_MINISTRAL_R01_CONTINUATION_READY"
    )
    return freeze


def build_freeze_v8(repository: Path = ROOT) -> dict:
    """Build the post-Qwen-r01 freeze through the versioned generator."""

    implementation_commit = "bdd32dcd49a95b82801f6b9e01b9111648237c78"
    notebook_pin_commit = "98d0d0df7a56af4f45375da69cc7c78072818f9d"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v7(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v8",
            "status": "FROZEN_AFTER_QWEN_R01_FOR_M4_BATCH_2_CONTINUATION",
            "reconciliation_date_utc": "2026-09-14",
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V7.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V7.json"
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
                "22-canonical/2-resource-gated/36-not-executed reconciliation and "
                "duplicate-prevention guard for subsequent M4-BATCH-2 execution."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_2_POST_QWEN_R01_CONTINUATION_READY"
    )
    return freeze


def build_freeze_v9(repository: Path = ROOT) -> dict:
    """Build the post-Llama-r02 M4-BATCH-3 freeze through the existing generator."""

    implementation_commit = "fc8f08ea76438159a7d3753066dc8c53146e5501"
    notebook_pin_commit = "32e6a89d2f529d2087b82f05557fb371e1ce6d0e"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v8(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v9",
            "status": "FROZEN_AFTER_LLAMA_R02_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-14",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V8.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V8.json"
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
                repository, implementation_commit, amendment
            ),
            "terminal_resource_gate_policy": {
                "contract_schema": "kaggle-vllm-m4-terminal-resource-gate-v1",
                "approved_reasons": ["VRAM_RESOURCE_GUARD"],
                "bare_return_code_3_is_sufficient": False,
                "cleanup_and_continuation_guards_required": True,
                "operational_and_integrity_failures_fail_stop": True,
            },
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "25-canonical/2-resource-gated/33-not-executed reconciliation and "
                "activates strict M4-BATCH-3 orchestration only for subsequent runs. "
                "The completed r02-llama execution remains V8/M4-BATCH-2 evidence."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_POST_LLAMA_R02_CONTINUATION_READY"
    )
    return freeze


def build_freeze_v10(repository: Path = ROOT) -> dict:
    """Build the post-Ministral-r02 M4-BATCH-3 freeze through the existing generator."""

    implementation_commit = "6b84c0d9a82336d6dacde7f9e334ef1e9898e2c3"
    notebook_pin_commit = "b6be3067ba454d5adc9e4906516618f38b04ef54"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v9(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v10",
            "status": "FROZEN_AFTER_MINISTRAL_R02_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-15",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V9.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V9.json"
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
                repository, implementation_commit, amendment
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "28-canonical/2-resource-gated/30-not-executed reconciliation for "
                "strict M4-BATCH-3 continuation. The completed r02-ministral execution "
                "remains immutable V9/M4-BATCH-3 evidence."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_POST_MINISTRAL_R02_CONTINUATION_READY"
    )
    return freeze


def build_freeze_v11(repository: Path = ROOT) -> dict:
    """Build the post-Qwen-r02 corrected M4-BATCH-3 continuation freeze."""

    implementation_commit = "c610a75b5d2612680fa44aa787928d558ad406ef"
    notebook_pin_commit = "01f70456acdbaf550f0ba3aea36b50fe8bf0b019"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v10(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v11",
            "status": "FROZEN_AFTER_QWEN_R02_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-16",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V10.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V10.json"
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
                repository, implementation_commit, amendment
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "28-canonical/3-resource-gated/29-not-executed reconciliation and "
                "the corrected real Qwen terminal-gate contract for future execution. "
                "The stopped r02-qwen attempt remains immutable V10/M4-BATCH-3 evidence."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_POST_QWEN_R02_CONTINUATION_READY"
    )
    return freeze


def build_freeze_v12(repository: Path = ROOT) -> dict:
    """Build the post-Qwen-r02-completion M4-BATCH-3 continuation freeze."""

    implementation_commit = "27fc9f3000246205e4a83fabc6244c7f4c5d98ff"
    notebook_pin_commit = "55d4bf61924729a297dbec787287a17d6d1b37d6"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v11(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v12",
            "status": "FROZEN_AFTER_QWEN_R02_COMPLETION_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-16",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V11.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V11.json"
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
                repository, implementation_commit, amendment
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "30-canonical/3-resource-gated/27-not-executed reconciliation and "
                "permanently closes r02-qwen against normal continuation. The V10 "
                "prefill-heavy attempt and V11 short/balanced continuation remain "
                "distinct immutable physical-session evidence."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_R02_QWEN_CLOSED_R02_PHI_READY"
    )
    return freeze


def build_freeze_v13(repository: Path = ROOT) -> dict:
    """Build the post-Phi-r02-completion M4-BATCH-3 continuation freeze."""

    implementation_commit = "571bd07f478f820813ff6c9b19faf3470b138fb5"
    notebook_pin_commit = "40c7c9da88bce712e3a87bcf9b5af52793cf6aa8"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v12(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v13",
            "status": "FROZEN_AFTER_PHI_R02_COMPLETION_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-19",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V12.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V12.json"
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
                repository, implementation_commit, amendment
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "33-canonical/3-resource-gated/24-not-executed reconciliation and "
                "permanently closes r02-phi against normal continuation. The V12 Phi "
                "batch remains immutable physical-session evidence."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_R02_PHI_CLOSED_R03_MINISTRAL_READY"
    )
    return freeze


def build_freeze_v14(repository: Path = ROOT) -> dict:
    """Build the post-Ministral-r03-completion M4-BATCH-3 continuation freeze."""

    implementation_commit = "e290855de8aed29570cdd7952f9c2b37a2a19443"
    notebook_pin_commit = "bd114f171f4ae9623b6e9538ed1789af90dce693"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v13(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v14",
            "status": "FROZEN_AFTER_MINISTRAL_R03_COMPLETION_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-19",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V13.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V13.json"
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
                repository, implementation_commit, amendment
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "36-canonical/3-resource-gated/21-not-executed reconciliation and "
                "permanently closes r03-ministral against normal continuation. The "
                "V13 Ministral batch and its exact reviewed post-execution notebook "
                "source-edit recovery remain immutable physical-session evidence."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_R03_MINISTRAL_CLOSED_R03_QWEN_READY"
    )
    return freeze


def build_freeze_v15(repository: Path = ROOT) -> dict:
    """Build the post-Qwen-r03-completion M4-BATCH-3 continuation freeze."""

    implementation_commit = "2ea82d306177aaca70066542c4bd6195e205d67e"
    notebook_pin_commit = "12e748a2c58db731b03dc28ba7477396cfe04f0f"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v14(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v15",
            "status": "FROZEN_AFTER_QWEN_R03_SETTLEMENT_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-19",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V14.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V14.json"
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
                repository, implementation_commit, amendment
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "38-canonical/4-resource-gated/18-not-executed reconciliation and "
                "permanently closes r03-qwen against normal continuation. The V14 "
                "Qwen short/balanced results and fourth independent prefill-heavy "
                "terminal resource result remain immutable physical-session evidence."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_R03_QWEN_CLOSED_R03_PHI_READY"
    )
    return freeze


def build_freeze_v16(repository: Path = ROOT) -> dict:
    """Build the post-Phi-r03-completion M4-BATCH-3 continuation freeze."""

    implementation_commit = "45e1da603e4f27dca67168a9d4058c50a6c60556"
    notebook_pin_commit = "da1395dd1c9e98968e3a29766069203be82aa4b6"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v15(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v16",
            "status": "FROZEN_AFTER_PHI_R03_COMPLETION_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-20",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V15.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V15.json"
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
                repository, implementation_commit, amendment
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "41-canonical/4-resource-gated/15-not-executed reconciliation and "
                "permanently closes r03-phi against normal continuation. The V15 "
                "Phi short, balanced, and prefill-heavy results remain immutable "
                "physical-session evidence."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_R03_PHI_CLOSED_R03_LLAMA_READY"
    )
    return freeze


def build_freeze_v17(repository: Path = ROOT) -> dict:
    """Build the post-Llama-r03-completion M4-BATCH-3 continuation freeze."""

    implementation_commit = "2f12aeb4a1753edfbae427af9d147f07b58a2189"
    notebook_pin_commit = "6a0637bc1dc3660ea9a3cb9918f951f5a444581f"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v16(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v17",
            "status": "FROZEN_AFTER_LLAMA_R03_COMPLETION_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-20",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V16.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V16.json"
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
                repository, implementation_commit, amendment
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "44-canonical/4-resource-gated/12-not-executed reconciliation and "
                "permanently closes r03-llama against normal continuation. The V16 "
                "Llama short, balanced, and prefill-heavy results remain immutable "
                "physical-session evidence. The required fifth Qwen prefill-heavy "
                "repetition remains queued without parameter or resource-limit change."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_R03_LLAMA_CLOSED_R04_QWEN_READY"
    )
    return freeze


def build_freeze_v18(repository: Path = ROOT) -> dict:
    """Build the post-Qwen-r04-completion M4-BATCH-3 continuation freeze."""

    implementation_commit = "0ea6fed84f8679ee449f39312fc8f61dde75762e"
    notebook_pin_commit = "9855590e758a7b70c9e4488f86af67b6fa3942eb"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v17(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v18",
            "status": "FROZEN_AFTER_QWEN_R04_COMPLETION_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-21",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V17.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V17.json"
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
                repository, implementation_commit, amendment
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "46-canonical/5-resource-gated/9-not-executed reconciliation and "
                "permanently closes r04-qwen against normal continuation. The V17 "
                "Qwen balanced/short results and fifth independent prefill-heavy "
                "terminal resource result remain immutable physical-session evidence."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_R04_QWEN_CLOSED_R04_PHI_READY"
    )
    return freeze


def build_freeze_v19(repository: Path = ROOT) -> dict:
    """Build the post-Phi-r04-completion M4-BATCH-3 continuation freeze."""

    implementation_commit = "4217f8c892c84e2f508d99cc11f1668e6bb5dd05"
    notebook_pin_commit = "6663f9e615d61a1218c72afd9060760c25e7ef07"
    amendment = "research/M4_BATCH_PROTOCOL_AMENDMENT_V3.md"
    notebook_relative = "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    notebook_blob = _blob(repository, notebook_pin_commit, notebook_relative)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
        temporary.write(notebook_blob)
        temporary.flush()
        source_digest = notebook_source_digest(Path(temporary.name))

    freeze = build_freeze_v18(repository)
    freeze.update(
        {
            "schema_version": "kaggle-vllm-m4-batch-source-freeze-v19",
            "status": "FROZEN_AFTER_PHI_R04_COMPLETION_FOR_M4_BATCH_3_CONTINUATION",
            "protocol_amendment_version": "M4-BATCH-3",
            "protocol_amendment_date_utc": "2026-09-14",
            "reconciliation_date_utc": "2026-09-21",
            "protocol_amendment_path": amendment,
            "implementation_source_commit": implementation_commit,
            "notebook_pin_commit": notebook_pin_commit,
            "historical_batch_2_freeze": {
                "path": "research/M4_BATCH_SOURCE_FREEZE_V18.json",
                "sha256": sha256_file(
                    repository / "research/M4_BATCH_SOURCE_FREEZE_V18.json"
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
                repository, implementation_commit, amendment
            ),
            "evidence_boundary": (
                "This freeze contains no new GPU measurements. It binds the reviewed "
                "49-canonical/5-resource-gated/6-not-executed reconciliation and "
                "permanently closes r04-phi against normal continuation. The V18 "
                "Phi balanced, prefill-heavy, and short results remain immutable "
                "physical-session evidence."
            ),
        }
    )
    freeze["execution_paths"]["batch_orchestrated"]["status"] = (
        "M4_BATCH_3_R04_PHI_CLOSED_R04_LLAMA_READY"
    )
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/M4_BATCH_SOURCE_FREEZE_V19.json",
    )
    parser.add_argument(
        "--render-notebook",
        action="store_true",
        help="render the canonical clean notebook from --implementation-commit",
    )
    parser.add_argument("--implementation-commit")
    args = parser.parse_args()
    if args.render_notebook:
        if not args.implementation_commit:
            parser.error("--render-notebook requires --implementation-commit")
        rendered = render_batch_notebook(ROOT, args.implementation_commit)
        print(json.dumps(rendered, indent=2))
        return 0
    args.output.write_text(
        json.dumps(build_freeze_v19(), indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
