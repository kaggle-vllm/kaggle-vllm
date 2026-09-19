"""Provenance-safe M4 batch guards and local batch ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .crossover import WORKLOAD_TOKENS
from .errors import ResearchEvidenceError
from .m4_ingest import (
    MAX_RUNTIME_TO_EXECUTION_DELAY_SECONDS,
    _check_existing_shard,
    _verify_runtime_binding,
    audit_download,
    inspect_zip,
    notebook_sources,
    verify_runtime,
)
from .provenance import sha256_file, verify_sha256_manifest
from .resources import GPU_MEMORY_LIMIT_MIB

BATCH_SCHEMA = "kaggle-vllm-m4-batch-manifest-v1"
BATCH_PLAN_SCHEMA = "kaggle-vllm-m4-batch-execution-plan-v1"
BATCH_PLAN_SCHEMA_V2 = "kaggle-vllm-m4-batch-execution-plan-v2"
OUTER_AUDIT_SCHEMA = "kaggle-vllm-m4-batch-ingest-audit-v1"
ALLOWED_OUTCOMES = {
    "PLANNED",
    "COMPLETED",
    "FAILED",
    "FAILED_RESOURCE_GATE",
    "NOT_EXECUTED",
    "SKIPPED_ALREADY_CANONICAL",
}
COMPLETED_WITH_TERMINAL_OUTCOMES = "COMPLETED_WITH_TERMINAL_OUTCOMES"
TERMINAL_RESOURCE_GATE_SCHEMA = "kaggle-vllm-m4-terminal-resource-gate-v1"
TERMINAL_RESOURCE_GATE_POLICY_VERSION = "M4-BATCH-3"
NOTEBOOK_RECOVERY_SCHEMA = "kaggle-vllm-m4-notebook-provenance-recovery-v1"
POST_EXECUTION_NOTEBOOK_SOURCE_EDIT = "POST_EXECUTION_NOTEBOOK_SOURCE_EDIT"
APPROVED_TERMINAL_RESOURCE_REASONS = {"VRAM_RESOURCE_GUARD"}
FROZEN_PRINCIPAL_CONCURRENCY = (1, 4, 8, 16, 32, 64)
HISTORICAL_STALE_NOTEBOOK_IDENTITY = {
    "schema_version": "kaggle-vllm-m4-batch-source-freeze-v7",
    "implementation_source_commit": "c05ca0db682074f29db9459d0cd9d50e162b34e6",
    "notebook_pin_commit": "b79961a7f4ff4e54ebf03ce917dafe12f8f191e5",
    "batch_notebook_source_digest": "7a0580043e7048bb66078c7b9f94cdbe56fefd5a35ff99ec032501f27020a800",
    "embedded_stale_digest": "03b4df794d7e61eb0842efac161db384ff679e67bb5bf93d9fe3156364178de6",
}
SESSION_ID = re.compile(r"^m4-[a-z0-9-]+-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
EMBEDDED_NOTEBOOK_SOURCE_DIGEST = re.compile(
    r"(?m)^\s*BATCH_NOTEBOOK_SOURCE_DIGEST\s*=\s*['\"]([0-9a-f]{64})['\"]\s*$"
)
MAX_OUTER_MEMBERS = 128
MAX_OUTER_MEMBER_BYTES = 1024**3
MAX_OUTER_UNCOMPRESSED_BYTES = 4 * 1024**3
TERMINAL_QUEUE_STATUSES = {
    "PRINCIPAL_SHARD_PRESERVED",
    "FAILED_RESOURCE_GATE",
    "FAILED_OTHER_REVIEW_REQUIRED",
}


def notebook_source_digest(path: Path) -> str:
    sources = [source for cell_type, _cell_id, source in notebook_sources(path) if cell_type == "code"]
    normalized = [
        re.sub(
            r"BATCH_NOTEBOOK_SOURCE_DIGEST = '[0-9a-f]{64}'",
            "BATCH_NOTEBOOK_SOURCE_DIGEST = '<NORMALIZED>'",
            source,
        )
        for source in sources
    ]
    payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_current_notebook_self_digest(path: Path) -> str:
    """Reject a newly prepared notebook whose unique embedded digest is stale."""

    matches = [
        match
        for _cell_type, _cell_id, source in notebook_sources(path)
        for match in EMBEDDED_NOTEBOOK_SOURCE_DIGEST.findall(source)
    ]
    expected = notebook_source_digest(path)
    if len(matches) != 1 or matches[0] != expected:
        raise ResearchEvidenceError(
            "current batch notebook must embed its recomputed source digest"
        )
    return expected


def _validate_notebook_source_identity(
    source_identity: dict[str, Any], freeze: dict[str, Any], frozen_notebook: Path
) -> str:
    """Bind a stale embedded self-digest only through the exact frozen source."""

    recorded = source_identity.get("batch_notebook_source_digest")
    expected = freeze.get("batch_notebook_source_digest")
    if recorded == expected:
        return "MATCH"
    matches = [
        match
        for _cell_type, _cell_id, source in notebook_sources(frozen_notebook)
        for match in EMBEDDED_NOTEBOOK_SOURCE_DIGEST.findall(source)
    ]
    if (
        len(matches) != 1
        or recorded != matches[0]
        or notebook_source_digest(frozen_notebook) != expected
        or freeze.get("schema_version")
        != HISTORICAL_STALE_NOTEBOOK_IDENTITY["schema_version"]
        or freeze.get("implementation_source_commit")
        != HISTORICAL_STALE_NOTEBOOK_IDENTITY["implementation_source_commit"]
        or freeze.get("notebook_pin_commit")
        != HISTORICAL_STALE_NOTEBOOK_IDENTITY["notebook_pin_commit"]
        or expected
        != HISTORICAL_STALE_NOTEBOOK_IDENTITY["batch_notebook_source_digest"]
        or recorded
        != HISTORICAL_STALE_NOTEBOOK_IDENTITY["embedded_stale_digest"]
    ):
        raise ResearchEvidenceError(
            "batch notebook source identity differs from freeze"
        )
    return "STALE_EMBEDDED_DIGEST_BOUND_BY_EXACT_FROZEN_SOURCE"


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchEvidenceError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ResearchEvidenceError(f"expected JSON object: {path}")
    return value


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_reviewed_notebook_recovery(
    *,
    repository: Path,
    recovery_path: Path,
    executed_notebook: Path,
    frozen_notebook: Path,
    batch_zip: Path,
    runtime_path: Path,
    manifest: dict[str, Any],
    freeze_path: Path,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    """Validate one explicit, reviewed post-execution notebook-source incident."""

    try:
        relative_recovery = recovery_path.resolve().relative_to(repository.resolve())
        relative_freeze = freeze_path.resolve().relative_to(repository.resolve())
    except ValueError as error:
        raise ResearchEvidenceError(
            "notebook recovery record must be repository-local"
        ) from error
    if relative_recovery.parts[:1] != ("research",):
        raise ResearchEvidenceError("notebook recovery record must be under research/")

    recovery = load_object(recovery_path)
    artifacts = recovery.get("artifacts", {})
    source = recovery.get("source_identity", {})
    changed = recovery.get("changed_cell", {})
    execution = recovery.get("execution_proof", {})
    if any(
        not isinstance(section, dict)
        for section in (artifacts, source, changed, execution)
    ):
        raise ResearchEvidenceError("notebook recovery record sections must be objects")

    checks = {
        "schema": recovery.get("schema_version") == NOTEBOOK_RECOVERY_SCHEMA,
        "classification": recovery.get("classification")
        == POST_EXECUTION_NOTEBOOK_SOURCE_EDIT,
        "approval": recovery.get("review_status")
        == "REVIEWED_APPROVED_FOR_EXACT_INGESTION",
        "batch": recovery.get("batch_id") == manifest.get("batch_id"),
        "session": recovery.get("session_id") == manifest.get("session_id"),
        "source commit": source.get("implementation_commit")
        == freeze.get("implementation_source_commit")
        == manifest.get("source_commit"),
        "notebook pin": source.get("notebook_pin_commit")
        == freeze.get("notebook_pin_commit"),
        "source freeze": source.get("source_freeze")
        == relative_freeze.as_posix(),
        "clean notebook hash": artifacts.get("clean_v13_notebook", {}).get(
            "sha256"
        )
        == sha256_file(frozen_notebook)
        == freeze.get("batch_notebook_sha256"),
        "clean source digest": source.get("clean_notebook_source_digest")
        == notebook_source_digest(frozen_notebook)
        == freeze.get("batch_notebook_source_digest"),
        "executed notebook hash": artifacts.get("saved_executed_notebook", {}).get(
            "sha256"
        )
        == sha256_file(executed_notebook),
        "outer ZIP hash": artifacts.get("outer_batch_zip", {}).get("sha256")
        == sha256_file(batch_zip),
        "runtime hash": artifacts.get("runtime", {}).get("sha256")
        == sha256_file(runtime_path),
    }
    if failed := [name for name, passed in checks.items() if not passed]:
        raise ResearchEvidenceError(
            f"notebook recovery identity mismatch: {failed}"
        )

    clean_sources = notebook_sources(frozen_notebook)
    executed_sources = notebook_sources(
        executed_notebook, allow_trailing_empty_code_cells=True
    )
    if len(clean_sources) != len(executed_sources):
        raise ResearchEvidenceError("notebook recovery cell count differs from freeze")
    differences = [
        index
        for index, (expected, observed) in enumerate(
            zip(clean_sources, executed_sources, strict=True)
        )
        if expected != observed
    ]
    if differences != [changed.get("index")]:
        raise ResearchEvidenceError(
            "notebook recovery must bind exactly one declared changed cell"
        )
    index = differences[0]
    expected_type, expected_id, expected_source = clean_sources[index]
    observed_type, observed_id, observed_source = executed_sources[index]
    cell_checks = {
        "code cell": expected_type == observed_type == "code",
        "cell id": expected_id == observed_id == changed.get("id"),
        "expected source": changed.get("expected_source") == expected_source,
        "altered source": changed.get("altered_source") == observed_source,
        "expected source hash": changed.get("expected_source_sha256")
        == hashlib.sha256(expected_source.encode()).hexdigest(),
        "altered source hash": changed.get("altered_source_sha256")
        == hashlib.sha256(observed_source.encode()).hexdigest(),
    }
    if failed := [name for name, passed in cell_checks.items() if not passed]:
        raise ResearchEvidenceError(
            f"notebook recovery changed-cell mismatch: {failed}"
        )

    notebook = load_object(executed_notebook)
    cells = notebook.get("cells", [])
    if not isinstance(cells, list) or index >= len(cells):
        raise ResearchEvidenceError("notebook recovery changed cell is absent")
    cell = cells[index]
    outputs = cell.get("outputs", []) if isinstance(cell, dict) else []
    metadata = cell.get("metadata", {}) if isinstance(cell, dict) else {}
    timing = metadata.get("execution", {}) if isinstance(metadata, dict) else {}
    if not isinstance(outputs, list) or not outputs or not isinstance(timing, dict):
        raise ResearchEvidenceError(
            "notebook recovery requires retained output and execution timing"
        )
    output_text = "".join(
        "".join(item.get("text", []))
        if isinstance(item.get("text"), list)
        else str(item.get("text", ""))
        for item in outputs
        if isinstance(item, dict)
    )
    output_checks = {
        "execution count": cell.get("execution_count")
        == execution.get("execution_count"),
        "output hash": execution.get("outputs_sha256") == _json_sha256(outputs),
        "batch output": f'"batch_id": "{manifest["batch_id"]}"' in output_text,
        "session output": f'"session_id": "{manifest["session_id"]}"'
        in output_text,
        "status output": '"status": "COMPLETED"' in output_text,
        "return code output": '"batch_runner_returncode": 0' in output_text,
        "outer hash output": artifacts["outer_batch_zip"]["sha256"] in output_text,
    }
    if failed := [name for name, passed in output_checks.items() if not passed]:
        raise ResearchEvidenceError(
            f"notebook recovery execution-output mismatch: {failed}"
        )

    started = str(timing.get("iopub.execute_input", ""))
    ended = str(timing.get("iopub.status.idle", ""))
    if (
        execution.get("cell_started_utc") != started
        or execution.get("cell_ended_utc") != ended
    ):
        raise ResearchEvidenceError("notebook recovery execution timing mismatch")
    try:
        cell_started = datetime.fromisoformat(started.replace("Z", "+00:00"))
        cell_ended = datetime.fromisoformat(ended.replace("Z", "+00:00"))
        batch_started = datetime.fromisoformat(str(manifest["start_utc"]))
        batch_ended = datetime.fromisoformat(str(manifest["end_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ResearchEvidenceError(
            "notebook recovery execution timestamps are invalid"
        ) from error
    if not cell_started <= batch_started <= batch_ended <= cell_ended:
        raise ResearchEvidenceError(
            "notebook recovery timing does not enclose the batch execution"
        )

    inner_hashes = {
        row["shard_id"]: row.get("evidence_zip_sha256")
        for row in manifest.get("shards", [])
        if row.get("evidence_zip_sha256")
    }
    if artifacts.get("inner_zip_sha256") != inner_hashes:
        raise ResearchEvidenceError("notebook recovery inner ZIP hashes mismatch")

    expected_runner = str(execution.get("expected_runner_path", ""))
    altered_runner = str(execution.get("altered_runner_path", ""))
    expected_blob = _git_blob(
        repository, str(freeze["implementation_source_commit"]), expected_runner
    )
    altered_blob = _git_blob(
        repository, str(freeze["implementation_source_commit"]), altered_runner
    )
    if (
        expected_runner != "scripts/kaggle_m4_execute_batch.py"
        or hashlib.sha256(expected_blob or b"").hexdigest()
        != freeze.get("batch_runner_sha256")
        or not altered_runner
        or altered_blob is not None
        or altered_runner not in observed_source
        or expected_runner not in expected_source
        or expected_source.count(expected_runner) != 1
        or observed_source != expected_source.replace(expected_runner, altered_runner)
    ):
        raise ResearchEvidenceError(
            "notebook recovery cannot prove the altered runner path was post-execution"
        )

    return {
        "classification": POST_EXECUTION_NOTEBOOK_SOURCE_EDIT,
        "record": relative_recovery.as_posix(),
        "record_sha256": sha256_file(recovery_path),
        "executed_notebook_sha256": sha256_file(executed_notebook),
        "frozen_notebook_sha256": sha256_file(frozen_notebook),
        "changed_cell_index": index,
        "changed_cell_id": expected_id,
    }


def select_batch(plan: dict[str, Any], batch_id: str) -> dict[str, Any]:
    if plan.get("schema_version") not in {BATCH_PLAN_SCHEMA, BATCH_PLAN_SCHEMA_V2}:
        raise ResearchEvidenceError("unsupported M4 batch plan schema")
    matches = [batch for batch in plan.get("batches", []) if batch.get("batch_id") == batch_id]
    if len(matches) != 1:
        raise ResearchEvidenceError(f"unknown or duplicate M4 batch ID: {batch_id}")
    batch = matches[0]
    repetitions = {item.get("repetition") for item in batch.get("execution_order", [])}
    if repetitions and repetitions != {batch.get("repetition")}:
        raise ResearchEvidenceError("batch mixes repetition indices")
    identities = {
        (item.get("model_key"), item.get("workload"))
        for item in batch.get("execution_order", [])
    }
    if len(identities) != len(batch.get("execution_order", [])):
        raise ResearchEvidenceError("batch duplicates a model/workload identity")
    return batch


def validate_batch_against_queue(
    batch: dict[str, Any], queue: dict[str, Any]
) -> dict[str, Any]:
    """Refuse stale batches that would execute any settled logical shard."""
    if queue.get("schema_version") != "kaggle-vllm-m4-principal-queue-v1":
        raise ResearchEvidenceError("unsupported M4 principal queue schema")
    if queue.get("active_shards") != 60 or queue.get("active_serving_cells") != 720:
        raise ResearchEvidenceError("principal queue does not preserve the 60-shard/720-cell design")
    active_rows = [row for row in queue.get("queue", []) if row.get("active_order") is not None]
    by_id = {row.get("shard_id"): row for row in active_rows}
    if len(active_rows) != 60 or len(by_id) != 60:
        raise ResearchEvidenceError("principal queue has missing or duplicate active shards")

    ordered = list(batch.get("ordered_shard_ids", []))
    execution_ids = [item.get("shard_id") for item in batch.get("execution_order", [])]
    if not ordered or ordered != execution_ids:
        raise ResearchEvidenceError("batch has zero remaining shards or inconsistent execution order")
    missing = [shard_id for shard_id in ordered if shard_id not in by_id]
    if missing:
        raise ResearchEvidenceError(f"batch references shards absent from active queue: {missing}")
    settled = [
        shard_id
        for shard_id in ordered
        if by_id[shard_id].get("status") in TERMINAL_QUEUE_STATUSES
    ]
    nonqueued = [
        shard_id
        for shard_id in ordered
        if by_id[shard_id].get("status") != "QUEUED" and shard_id not in settled
    ]
    if settled:
        raise ResearchEvidenceError(f"refusing to reschedule settled M4 shards: {settled}")
    if nonqueued:
        raise ResearchEvidenceError(f"batch contains non-queued M4 shards: {nonqueued}")

    preserved_skips = list(batch.get("already_completed_skips", []))
    review_exclusions = list(batch.get("review_required_exclusions", []))
    if any(by_id.get(shard_id, {}).get("status") != "PRINCIPAL_SHARD_PRESERVED" for shard_id in preserved_skips):
        raise ResearchEvidenceError("batch canonical-skip state differs from principal queue")
    if any(by_id.get(shard_id, {}).get("status") not in {"FAILED_RESOURCE_GATE", "FAILED_OTHER_REVIEW_REQUIRED"} for shard_id in review_exclusions):
        raise ResearchEvidenceError("batch review-required exclusion differs from principal queue")
    return {
        "status": "PASS_NO_SETTLED_SHARD_RESCHEDULED",
        "queued_shard_ids": ordered,
        "settled_exclusions": [*preserved_skips, *review_exclusions],
    }


def check_disk_capacity(
    path: Path,
    *,
    projected_additional_bytes: int,
    reserve_bytes: int,
    usage: Callable[[Path], Any] = shutil.disk_usage,
) -> dict[str, int]:
    disk = usage(path)
    required = projected_additional_bytes + reserve_bytes
    report = {
        "total_bytes": int(disk.total),
        "used_bytes": int(disk.used),
        "free_bytes": int(disk.free),
        "projected_additional_bytes": projected_additional_bytes,
        "reserve_bytes": reserve_bytes,
        "required_free_bytes": required,
    }
    if disk.free < required:
        raise ResearchEvidenceError(
            f"insufficient disk before shard: {disk.free} free bytes < {required} required bytes"
        )
    return report


def check_wall_clock(
    elapsed_seconds: float,
    *,
    maximum_seconds: int,
    minimum_remaining_seconds: int,
) -> dict[str, float | int]:
    remaining = maximum_seconds - elapsed_seconds
    report = {
        "elapsed_seconds": elapsed_seconds,
        "maximum_seconds": maximum_seconds,
        "remaining_seconds": remaining,
        "minimum_remaining_seconds": minimum_remaining_seconds,
    }
    if remaining < minimum_remaining_seconds:
        raise ResearchEvidenceError("batch wall-clock safety limit reached before next shard")
    return report


def terminal_resource_continuation_enabled(
    plan: dict[str, Any], batch: dict[str, Any]
) -> bool:
    """Require an explicit prospective protocol opt-in; V1/V2 remain fail-stop."""

    policy = plan.get("terminal_resource_gate_policy")
    return bool(
        isinstance(policy, dict)
        and policy.get("enabled") is True
        and policy.get("protocol_amendment_version")
        == TERMINAL_RESOURCE_GATE_POLICY_VERSION
        and policy.get("contract_schema") == TERMINAL_RESOURCE_GATE_SCHEMA
        and policy.get("approved_reasons")
        == sorted(APPROVED_TERMINAL_RESOURCE_REASONS)
        and batch.get("protocol_amendment_version")
        == TERMINAL_RESOURCE_GATE_POLICY_VERSION
    )


def exact_hf_repo_cache_path(hf_home: Path, hf_id: str) -> Path:
    parts = hf_id.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ResearchEvidenceError(f"unsafe Hugging Face repository ID: {hf_id}")
    hub = (hf_home / "hub").resolve()
    candidate = (hub / f"models--{parts[0]}--{parts[1]}").resolve()
    if candidate.parent != hub or not candidate.name.startswith("models--"):
        raise ResearchEvidenceError("cannot identify exact Hugging Face model cache path")
    return candidate


def _resource_ledger_audit(evidence: Path) -> dict[str, Any]:
    peaks = {0: 0.0, 1: 0.0}
    sample_counts = {0: 0, 1: 0}
    cell_peaks: dict[str, dict[int, float]] = {}
    ledgers = sorted(evidence.glob("*.resources.jsonl")) + sorted(
        evidence.glob("*.telemetry.jsonl")
    )
    if not ledgers:
        raise ResearchEvidenceError("shard has no resource or telemetry ledgers")
    for ledger in ledgers:
        cell = ledger.name.removesuffix(".resources.jsonl").removesuffix(
            ".telemetry.jsonl"
        )
        cell_peaks.setdefault(cell, {0: 0.0, 1: 0.0})
        for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
            try:
                row = json.loads(line)
                gpu = row["gpu_index"] if "gpu_index" in row else row["index"]
                memory = float(row["memory_used_mib"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ResearchEvidenceError(
                    f"invalid GPU resource row: {ledger}:{line_number}"
                ) from error
            if gpu not in peaks or memory < 0:
                raise ResearchEvidenceError(f"invalid physical GPU sample: {ledger}:{line_number}")
            peaks[gpu] = max(peaks[gpu], memory)
            cell_peaks[cell][gpu] = max(cell_peaks[cell][gpu], memory)
            sample_counts[gpu] += 1
    if any(count == 0 for count in sample_counts.values()):
        raise ResearchEvidenceError("resource evidence does not cover both physical GPUs")
    return {
        "gpu0_maximum_observed_mib": peaks[0],
        "gpu1_maximum_observed_mib": peaks[1],
        "per_gpu_limit_mib": GPU_MEMORY_LIMIT_MIB,
        "status": (
            "VRAM_RESOURCE_GUARD"
            if any(peak > GPU_MEMORY_LIMIT_MIB for peak in peaks.values())
            else "PASS"
        ),
        "offending_physical_gpus": [
            {"index": gpu, "maximum_observed_mib": peak}
            for gpu, peak in sorted(peaks.items())
            if peak > GPU_MEMORY_LIMIT_MIB
        ],
        "offending_cells": [
            {
                "cell": cell,
                "physical_gpus": [
                    {"index": gpu, "maximum_observed_mib": peak}
                    for gpu, peak in sorted(gpu_peaks.items())
                    if peak > GPU_MEMORY_LIMIT_MIB
                ],
            }
            for cell, gpu_peaks in sorted(cell_peaks.items())
            if any(peak > GPU_MEMORY_LIMIT_MIB for peak in gpu_peaks.values())
        ],
    }


def measure_shard_peaks(evidence: Path) -> dict[str, Any]:
    audit = _resource_ledger_audit(evidence)
    if audit["status"] != "PASS":
        first = audit["offending_physical_gpus"][0]
        raise ResearchEvidenceError(
            f"per-GPU VRAM limit exceeded: GPU{first['index']} "
            f"{first['maximum_observed_mib']} MiB > {GPU_MEMORY_LIMIT_MIB} MiB"
        )
    return audit


def verify_terminal_resource_gate(
    evidence: Path,
    *,
    expected_shard: dict[str, Any],
    expected_model: dict[str, Any],
    expected_source_commit: str,
    runtime: dict[str, Any],
    maximum_runtime_delay_seconds: float,
) -> dict[str, Any]:
    """Verify the sole continuable scientific failure contract, fail closed."""

    verify_sha256_manifest(evidence, evidence / "SHA256SUMS.txt")
    runner_manifest = evidence / "RUNNER_SHA256SUMS.txt"
    if runner_manifest.is_file():
        verify_sha256_manifest(evidence, runner_manifest)
    start = load_object(evidence / "execution-start.json")
    summary = load_object(evidence / "execution-summary.json")
    raw = load_object(evidence / "m4-raw.json")
    gate = load_object(evidence / "terminal-resource-gate.json")
    prompt = load_object(evidence / "prompt-manifest.json")
    _verify_runtime_binding(
        runtime,
        start,
        maximum_delay_seconds=maximum_runtime_delay_seconds,
    )

    expected_shard_id = expected_shard.get("shard_id")
    source = start.get("source", {})
    checks = {
        "gate schema": gate.get("schema_version") == TERMINAL_RESOURCE_GATE_SCHEMA,
        "classification": gate.get("classification") == "FAILED_RESOURCE_GATE",
        "approved reason": gate.get("reason") in APPROVED_TERMINAL_RESOURCE_REASONS,
        "runner return code": gate.get("runner_returncode") == 3,
        "shard ID": gate.get("shard_id") == expected_shard_id,
        "source commit": source.get("commit") == expected_source_commit
        and source.get("dirty") is False
        and raw.get("source") == source
        and gate.get("source_commit") == expected_source_commit,
        "principal mode": start.get("mode") == raw.get("mode") == "principal",
        "model": start.get("model_key")
        == raw.get("model_key")
        == expected_shard.get("model_key"),
        "workload": start.get("workload")
        == raw.get("workload")
        == expected_shard.get("workload"),
        "repetition": start.get("repetition")
        == raw.get("repetition")
        == expected_shard.get("repetition"),
        "fresh server": raw.get("server_lifecycle") == "fresh_server_per_cell",
        "SDK version": start.get("sdk_version") == "0.2.0",
        "model identity": start.get("model", {}).get("hf_id")
        == raw.get("model_id")
        == expected_model.get("hf_id")
        and start.get("model", {}).get("revision")
        == raw.get("model_revision")
        == expected_model.get("revision"),
        "hard threshold": start.get("resource_limits", {}).get(
            "per_gpu_memory_mib"
        )
        == GPU_MEMORY_LIMIT_MIB
        and gate.get("per_gpu_limit_mib") == GPU_MEMORY_LIMIT_MIB,
        "resource status": summary.get("status")
        == raw.get("status")
        == "RESOURCE_GUARD_VIOLATION",
        "resource message": isinstance(raw.get("resource_guard_violation"), str)
        and raw.get("resource_guard_violation")
        == summary.get("resource_guard_violation")
        == gate.get("resource_guard_violation"),
        "monitor action": gate.get("monitor_action")
        == "TERMINATE_CELL_PROCESS_GROUP",
        "no semantic failure": summary.get("semantic_gate_failure") is None
        and raw.get("semantic_gate_failure") is None,
        "complete terminal grid": summary.get("expected_cells") == 12
        and summary.get("completed_or_preserved_cells") == 12
        and summary.get("failed_cells") == 1,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ResearchEvidenceError(
            f"terminal resource gate contract mismatch: {failed_checks}"
        )

    expected_tokens = WORKLOAD_TOKENS.get(str(expected_shard.get("workload")))
    prompts = prompt.get("prompts")
    prompt_sha256 = sha256_file(evidence / "prompt-manifest.json")
    if (
        expected_tokens is None
        or prompt.get("model_id") != expected_model.get("hf_id")
        or prompt.get("model_revision") != expected_model.get("revision")
        or prompt.get("target_input_tokens") != expected_tokens[0]
        or not isinstance(prompts, list)
        or len(prompts) != 64
        or any(item.get("token_count") != expected_tokens[0] for item in prompts)
        or raw.get("prompt_manifest_sha256") != prompt_sha256
    ):
        raise ResearchEvidenceError(
            "terminal resource gate prompt/model identity differs from the frozen workload"
        )

    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ResearchEvidenceError("terminal resource gate lacks the complete raw grid")
    identities = {
        (row.get("tensor_parallel_size"), row.get("concurrency"))
        for row in rows
        if isinstance(row, dict)
    }
    expected_identities = {
        (tp, concurrency)
        for concurrency in FROZEN_PRINCIPAL_CONCURRENCY
        for tp in (1, 2)
    }
    if identities != expected_identities or any(row.get("oom") is not False for row in rows):
        raise ResearchEvidenceError(
            "terminal resource gate grid is incomplete or reports CUDA OOM"
        )
    if any(
        row.get("model_id") != expected_model.get("hf_id")
        or row.get("model_revision") != expected_model.get("revision")
        or row.get("workload") != expected_shard.get("workload")
        or row.get("repetition") != expected_shard.get("repetition")
        or row.get("input_tokens") != expected_tokens[0]
        or row.get("output_tokens_requested") != expected_tokens[1]
        or row.get("prompt_manifest_sha256") != prompt_sha256
        for row in rows
    ):
        raise ResearchEvidenceError(
            "terminal resource gate row identity differs from the frozen workload"
        )

    ledger = _resource_ledger_audit(evidence)
    if ledger["status"] != "VRAM_RESOURCE_GUARD" or len(ledger["offending_cells"]) != 1:
        raise ResearchEvidenceError(
            "terminal resource gate is not supported by one offending cell ledger"
        )
    offending = ledger["offending_cells"][0]
    expected_cell = gate.get("failed_cell")
    if offending["cell"] != expected_cell:
        raise ResearchEvidenceError("terminal resource gate failed-cell identity mismatch")
    if gate.get("offending_physical_gpus") != offending["physical_gpus"]:
        raise ResearchEvidenceError("terminal resource gate physical-GPU evidence mismatch")

    match = re.fullmatch(
        rf"{re.escape(str(expected_shard_id))}-tp([12])-c(01|04|08|16|32|64)",
        str(expected_cell),
    )
    if match is None:
        raise ResearchEvidenceError("terminal resource gate has invalid failed-cell identity")
    identity = (int(match.group(1)), int(match.group(2)))
    failed_row = next(
        row
        for row in rows
        if (row.get("tensor_parallel_size"), row.get("concurrency")) == identity
    )
    null_metrics = (
        "request_throughput_per_second",
        "input_tokens_per_second",
        "output_tokens_per_second",
        "total_tokens_per_second",
        "ttft_ms",
        "tpot_ms",
        "itl_ms",
        "e2e_latency_ms",
    )
    if (
        failed_row.get("request_failures", 0) <= 0
        or any(failed_row.get(metric) is not None for metric in null_metrics)
        or failed_row.get("maximum_vram_mib", 0) <= GPU_MEMORY_LIMIT_MIB
    ):
        raise ResearchEvidenceError(
            "terminal resource gate failed row has invalid missing-measurement semantics"
        )

    cell = load_object(evidence / f"{expected_cell}.json")
    raw_observations = cell.get("failure_observations")
    observations = set(raw_observations) if isinstance(raw_observations, list) else set()
    measurements = cell.get("measurements", {})
    failed_requests = measurements.get("failed_requests")
    if (
        cell.get("status") != "executed"
        or cell.get("oom_observed") is not False
        or cell.get("server", {}).get("unexpected_exit_returncode") != 0
        or not isinstance(raw_observations, list)
        or observations != {"connection_error", "server_exit"}
        or failed_requests != 192
        or measurements.get("successful_requests") != 0
        or measurements.get("failure_counts") != {"connection_error": failed_requests}
        or measurements.get("requests", {}).get("count") != failed_requests
        or measurements.get("input_tokens") != 0
        or measurements.get("output_tokens") != 0
        or cell.get("identity", {}).get("source_git_commit")
        != expected_source_commit
        or cell.get("engine", {}).get("model") != expected_model.get("hf_id")
        or cell.get("engine", {}).get("model_revision")
        != expected_model.get("revision")
        or cell.get("engine", {}).get("dtype") != "float16"
        or cell.get("engine", {}).get("tensor_parallel_size") != identity[0]
        or cell.get("engine", {}).get("gpu_memory_utilization") != 0.9
    ):
        raise ResearchEvidenceError(
            "terminal resource gate includes an unapproved operational cell failure"
        )
    log_path = evidence / str(cell.get("server", {}).get("server_log", ""))
    log = log_path.read_text(encoding="utf-8", errors="replace").casefold()
    disallowed = (
        "cuda out of memory",
        "outofmemoryerror",
        "nccl error",
        "ncclerror",
        "failed to load model",
        "error loading model",
        "engine core initialization failed",
    )
    if any(marker in log for marker in disallowed):
        raise ResearchEvidenceError(
            "terminal resource gate includes CUDA OOM, NCCL, or model-load evidence"
        )
    if (evidence / "compatibility-gate.json").exists() or (
        evidence / "semantic-gate-failure.json"
    ).exists():
        raise ResearchEvidenceError(
            "terminal resource gate includes an operational or semantic gate failure"
        )
    return {
        "status": "VERIFIED_TERMINAL_RESOURCE_GATE",
        "classification": "FAILED_RESOURCE_GATE",
        "reason": gate["reason"],
        "failed_cell": expected_cell,
        "per_gpu_limit_mib": GPU_MEMORY_LIMIT_MIB,
        "offending_physical_gpus": offending["physical_gpus"],
        "gpu0_maximum_observed_mib": ledger["gpu0_maximum_observed_mib"],
        "gpu1_maximum_observed_mib": ledger["gpu1_maximum_observed_mib"],
        "failed_cell_throughput": None,
        "oom_observed": False,
    }


def inspect_batch_zip(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_OUTER_MEMBERS:
                raise ResearchEvidenceError("M4 batch ZIP has too many members")
            names: set[str] = set()
            total = 0
            for info in infos:
                name = info.filename
                relative = PurePosixPath(name)
                mode = info.external_attr >> 16
                if (
                    not name
                    or "\\" in name
                    or "\x00" in name
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or name != relative.as_posix()
                    or len(relative.parts) != 1
                    or info.is_dir()
                ):
                    raise ResearchEvidenceError(f"unsafe batch ZIP path: {name}")
                if name in names:
                    raise ResearchEvidenceError(f"duplicate batch ZIP member: {name}")
                if info.flag_bits & 0x1:
                    raise ResearchEvidenceError(f"encrypted batch ZIP member: {name}")
                if stat.S_ISLNK(mode):
                    raise ResearchEvidenceError(f"symlink batch ZIP member: {name}")
                if info.file_size > MAX_OUTER_MEMBER_BYTES:
                    raise ResearchEvidenceError(f"oversized batch ZIP member: {name}")
                total += info.file_size
                if total > MAX_OUTER_UNCOMPRESSED_BYTES:
                    raise ResearchEvidenceError("M4 batch ZIP exceeds size limit")
                names.add(name)
            bad_crc = archive.testzip()
    except (OSError, zipfile.BadZipFile) as error:
        raise ResearchEvidenceError(f"invalid batch ZIP: {path}") from error
    if bad_crc is not None:
        raise ResearchEvidenceError(f"batch ZIP CRC failure: {bad_crc}")
    required = {
        "BATCH_MANIFEST.json",
        "BATCH_SHA256SUMS.txt",
        "BATCH_SOURCE_IDENTITY.json",
        "batch.log",
        "runtime.json",
    }
    missing = required - names
    if missing:
        raise ResearchEvidenceError(f"batch ZIP lacks required members: {sorted(missing)}")
    return sorted(names)


def _git_blob(repository: Path, commit: str, relative: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    return completed.stdout if completed.returncode == 0 else None


def _is_git_worktree(repository: Path) -> bool:
    completed = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == "true"


def _load_frozen_object(
    repository: Path,
    *,
    commit: str,
    relative: str,
    expected_sha256: str,
) -> dict[str, Any]:
    blob = _git_blob(repository, commit, relative)
    if blob is not None:
        if hashlib.sha256(blob).hexdigest() != expected_sha256:
            raise ResearchEvidenceError(f"frozen source hash mismatch: {relative}")
        try:
            value = json.loads(blob)
        except json.JSONDecodeError as error:
            raise ResearchEvidenceError(
                f"frozen source is not valid JSON: {relative}"
            ) from error
        if not isinstance(value, dict):
            raise ResearchEvidenceError(f"frozen source is not an object: {relative}")
        return value
    current = repository / relative
    if not current.is_file() or sha256_file(current) != expected_sha256:
        raise ResearchEvidenceError(
            f"cannot resolve hash-matching frozen source: {relative}"
        )
    return load_object(current)


def verify_batch_source_freeze(
    repository: Path,
    freeze_path: Path | None = None,
) -> dict[str, Any]:
    freeze_path = freeze_path or repository / "research/M4_BATCH_SOURCE_FREEZE.json"
    freeze = load_object(freeze_path)
    batch_plan_path = freeze.get(
        "batch_plan_path", "research/M4_BATCH_EXECUTION_PLAN.json"
    )
    amendment_path = freeze.get(
        "protocol_amendment_path", "research/M4_BATCH_PROTOCOL_AMENDMENT.md"
    )
    paths = {
        "batch_notebook_sha256": "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb",
        "batch_runner_sha256": "scripts/kaggle_m4_execute_batch.py",
        "base_shard_runner_sha256": "scripts/kaggle_m4_multimodel_crossover.py",
        "m4_execution_plan_sha256": "research/M4_EXECUTION_PLAN.json",
        "model_matrix_sha256": "research/model_matrix.json",
        "protocol_sha256": "research/m4_protocol.json",
        "batch_plan_sha256": batch_plan_path,
        "protocol_amendment_sha256": amendment_path,
    }
    if "principal_queue_sha256" in freeze:
        paths["principal_queue_sha256"] = freeze.get(
            "principal_queue_path", "research/M4_PRINCIPAL_EXECUTION_QUEUE.json"
        )
    implementation_commit = str(freeze.get("implementation_source_commit", ""))
    notebook_commit = str(freeze.get("notebook_pin_commit", ""))
    is_git_worktree = _is_git_worktree(repository)
    try:
        freeze_relative = freeze_path.resolve().relative_to(repository.resolve()).as_posix()
    except ValueError:
        freeze_relative = ""
    head_freeze = (
        _git_blob(repository, "HEAD", freeze_relative)
        if is_git_worktree and freeze_relative
        else None
    )
    tracked_freeze_is_clean = head_freeze == freeze_path.read_bytes()
    if head_freeze is not None and not tracked_freeze_is_clean:
        raise ResearchEvidenceError("M4 batch source-freeze record differs from HEAD")
    for field, relative in paths.items():
        commit = notebook_commit if field == "batch_notebook_sha256" else implementation_commit
        blob = _git_blob(repository, commit, relative)
        if blob is None and tracked_freeze_is_clean:
            recorded = freeze.get(field)
            if not isinstance(recorded, str) or re.fullmatch(r"[0-9a-f]{64}", recorded) is None:
                raise ResearchEvidenceError(f"invalid M4 batch source-freeze hash: {field}")
            continue
        digest = hashlib.sha256(blob).hexdigest() if blob is not None else sha256_file(repository / relative)
        if freeze.get(field) != digest:
            raise ResearchEvidenceError(f"M4 batch source-freeze mismatch: {relative}")
    commit = implementation_commit
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ResearchEvidenceError("invalid batch implementation source commit")
    if re.fullmatch(r"[0-9a-f]{40}", notebook_commit) is None:
        raise ResearchEvidenceError("invalid batch notebook pin commit")
    notebook_blob = _git_blob(
        repository, notebook_commit, "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    )
    if notebook_blob is None and tracked_freeze_is_clean:
        recorded_source_digest = freeze.get("batch_notebook_source_digest")
        if not isinstance(recorded_source_digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", recorded_source_digest
        ) is None:
            raise ResearchEvidenceError("invalid M4 batch notebook source digest")
        return freeze
    if notebook_blob is None:
        notebook_path = repository / "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
        source_digest = notebook_source_digest(notebook_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".ipynb") as temporary:
            temporary.write(notebook_blob)
            temporary.flush()
            source_digest = notebook_source_digest(Path(temporary.name))
    if freeze.get("batch_notebook_source_digest") != source_digest:
        raise ResearchEvidenceError("M4 batch notebook source digest mismatch")
    return freeze


def select_batch_source_freeze(repository: Path, source_commit: str) -> tuple[Path, dict[str, Any]]:
    candidates = sorted(repository.glob("research/M4_BATCH_SOURCE_FREEZE*.json"))
    matches = [
        path
        for path in candidates
        if load_object(path).get("implementation_source_commit") == source_commit
    ]
    if len(matches) != 1:
        raise ResearchEvidenceError(
            f"unknown or ambiguous M4 batch source freeze for commit: {source_commit}"
        )
    return matches[0], verify_batch_source_freeze(repository, matches[0])


def validate_batch_manifest(
    manifest: dict[str, Any],
    batch: dict[str, Any],
    *,
    expected_source_commit: str,
    allow_terminal_resource_continuation: bool = False,
) -> None:
    if manifest.get("schema_version") != BATCH_SCHEMA:
        raise ResearchEvidenceError("unsupported M4 batch manifest schema")
    checks = {
        "batch ID": manifest.get("batch_id") == batch.get("batch_id"),
        "repetition": manifest.get("repetition") == batch.get("repetition"),
        "execution mode": manifest.get("execution_mode") == "batch_orchestrated",
        "source commit": manifest.get("source_commit") == expected_source_commit,
        "planned order": manifest.get("ordered_shard_ids") == batch.get("ordered_shard_ids"),
        "canonical skips": manifest.get("already_completed_skips")
        == batch.get("already_completed_skips"),
        "session ID": isinstance(manifest.get("session_id"), str)
        and SESSION_ID.fullmatch(manifest["session_id"]) is not None,
    }
    if plan_version := batch.get("protocol_amendment_version"):
        checks["protocol amendment"] = (
            manifest.get("protocol_amendment_version") == plan_version
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ResearchEvidenceError(f"batch manifest identity mismatch: {failed}")
    outcomes = manifest.get("shards")
    if not isinstance(outcomes, list):
        raise ResearchEvidenceError("batch manifest shard outcomes must be a list")
    ids = [item.get("shard_id") for item in outcomes if isinstance(item, dict)]
    expected_ids = batch.get("already_completed_skips", []) + batch.get(
        "ordered_shard_ids", []
    )
    if len(ids) != len(outcomes) or len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ResearchEvidenceError("batch manifest has missing or duplicate shard IDs")
    outcome_by_id = {item["shard_id"]: item for item in outcomes}
    for shard_id in batch.get("already_completed_skips", []):
        if outcome_by_id[shard_id].get("status") != "SKIPPED_ALREADY_CANONICAL":
            raise ResearchEvidenceError("canonical skip status mismatch")
    started = []
    terminal_seen = False
    for order, shard_id in enumerate(batch.get("ordered_shard_ids", []), 1):
        outcome = outcome_by_id[shard_id]
        status_value = outcome.get("status")
        if status_value not in ALLOWED_OUTCOMES:
            raise ResearchEvidenceError(f"unknown batch shard outcome: {status_value}")
        if outcome.get("within_session_order") != order:
            raise ResearchEvidenceError("within-session execution order mismatch")
        if status_value in {"COMPLETED", "FAILED", "FAILED_RESOURCE_GATE"}:
            if terminal_seen:
                raise ResearchEvidenceError("batch execution order is not a prefix")
            started.append(shard_id)
            if status_value == "FAILED":
                terminal_seen = True
            elif status_value == "FAILED_RESOURCE_GATE":
                if not allow_terminal_resource_continuation:
                    raise ResearchEvidenceError(
                        "terminal resource continuation is not authorized by the frozen plan"
                    )
                terminal = outcome.get("terminal_resource_gate", {})
                cleanup = outcome.get("post_shard_gpu_cleanup", {})
                if (
                    outcome.get("runner_returncode") != 3
                    or outcome.get("classification") != "FAILED_RESOURCE_GATE"
                    or outcome.get("reason") not in APPROVED_TERMINAL_RESOURCE_REASONS
                    or terminal.get("status") != "VERIFIED_TERMINAL_RESOURCE_GATE"
                    or terminal.get("classification") != "FAILED_RESOURCE_GATE"
                    or terminal.get("reason") != outcome.get("reason")
                    or cleanup.get("status") != "PASS"
                ):
                    raise ResearchEvidenceError(
                        "batch manifest has an unverified terminal resource outcome"
                    )
        elif status_value == "NOT_EXECUTED":
            terminal_seen = True
        else:
            raise ResearchEvidenceError("planned batch row lacks a terminal outcome")
        if status_value in {"COMPLETED", "FAILED_RESOURCE_GATE"} and (
            not outcome.get("evidence_zip") or not outcome.get("evidence_zip_sha256")
        ):
            raise ResearchEvidenceError("settled shard lacks ZIP identity")
    if manifest.get("actual_execution_order") != started:
        raise ResearchEvidenceError("actual execution order differs from shard outcomes")
    if allow_terminal_resource_continuation:
        planned = [outcome_by_id[shard_id] for shard_id in batch["ordered_shard_ids"]]
        resource_count = sum(
            item.get("status") == "FAILED_RESOURCE_GATE" for item in planned
        )
        failed_count = sum(item.get("status") == "FAILED" for item in planned)
        not_executed_count = sum(
            item.get("status") == "NOT_EXECUTED" for item in planned
        )
        if (
            resource_count
            and not failed_count
            and not not_executed_count
            and (
                manifest.get("status") != COMPLETED_WITH_TERMINAL_OUTCOMES
                or manifest.get("stop_reason") is not None
            )
        ):
            raise ResearchEvidenceError(
                "completed terminal-outcome batch status is inconsistent"
            )


def _validate_batch_shard_provenance(
    provenance: dict[str, Any],
    *,
    shard_id: str,
    outcome: dict[str, Any],
    manifest: dict[str, Any],
    expected_source_commit: str,
) -> None:
    checks = {
        "schema": provenance.get("schema_version")
        == "kaggle-vllm-m4-batch-shard-provenance-v1",
        "mode": provenance.get("execution_mode") == "batch_orchestrated",
        "shard": provenance.get("shard_id") == shard_id,
        "batch": provenance.get("batch_id") == manifest["batch_id"],
        "session": provenance.get("session_id") == manifest["session_id"],
        "repetition": provenance.get("repetition") == manifest["repetition"],
        "order": provenance.get("within_session_order")
        == outcome["within_session_order"],
        "source": provenance.get("source_commit") == expected_source_commit,
        "return code": provenance.get("runner_returncode")
        == outcome.get("runner_returncode"),
    }
    if failed := [name for name, passed in checks.items() if not passed]:
        raise ResearchEvidenceError(
            f"inner batch-shard provenance mismatch: {failed}"
        )


def _audit_terminal_resource_archive(
    *,
    evidence_zip: Path,
    runtime: dict[str, Any],
    expected_shard: dict[str, Any],
    expected_model: dict[str, Any],
    outcome: dict[str, Any],
    manifest: dict[str, Any],
    expected_source_commit: str,
    maximum_runtime_delay_seconds: float,
) -> dict[str, Any]:
    members = inspect_zip(evidence_zip)
    extracted = Path(tempfile.mkdtemp(prefix="kaggle-vllm-m4-resource-gate-"))
    try:
        with zipfile.ZipFile(evidence_zip) as archive:
            archive.extractall(extracted)
        verified = verify_sha256_manifest(
            extracted, extracted / "SHA256SUMS.txt"
        )
        if set(verified) != set(members) - {"SHA256SUMS.txt"}:
            raise ResearchEvidenceError(
                "terminal resource SHA256SUMS.txt must cover every payload exactly"
            )
        terminal = verify_terminal_resource_gate(
            extracted,
            expected_shard=expected_shard,
            expected_model=expected_model,
            expected_source_commit=expected_source_commit,
            runtime=runtime,
            maximum_runtime_delay_seconds=maximum_runtime_delay_seconds,
        )
        provenance = load_object(extracted / "batch-shard-provenance.json")
        _validate_batch_shard_provenance(
            provenance,
            shard_id=str(expected_shard["shard_id"]),
            outcome=outcome,
            manifest=manifest,
            expected_source_commit=expected_source_commit,
        )
        return {
            **terminal,
            "shard_id": expected_shard["shard_id"],
            "evidence_zip_sha256": sha256_file(evidence_zip),
            "preservation_status": "PRESERVED_IN_ORIGINAL_OUTER_BATCH_FOR_REVIEW",
        }
    finally:
        shutil.rmtree(extracted, ignore_errors=True)


def _stage_inner(
    *,
    repository: Path,
    extracted: Path,
    audit: dict[str, Any],
    notebook: Path,
    inner_zip: Path,
    runtime_path: Path,
    provenance: dict[str, Any],
    recovery_record_path: Path | None = None,
) -> None:
    destination = repository / ".local-evidence/m4-ingest" / audit["evidence_zip_sha256"]
    _check_existing_shard(repository, audit["shard_id"], destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    originals = extracted / "original"
    originals.mkdir()
    for source in (notebook, inner_zip, runtime_path):
        shutil.copy2(source, originals / source.name)
    if recovery_record_path is not None:
        shutil.copy2(recovery_record_path, originals / recovery_record_path.name)
    audit.update(provenance)
    audit["staged_at"] = str(destination)
    (extracted / "INGEST_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    shutil.move(str(extracted), destination)


def stage_batch_download(
    *,
    repository: Path,
    notebook: Path,
    batch_zip: Path,
    runtime_path: Path,
    notebook_recovery_record: Path | None = None,
) -> dict[str, Any]:
    members = inspect_batch_zip(batch_zip)
    runtime = load_object(runtime_path)
    verify_runtime(runtime)
    temporary = Path(tempfile.mkdtemp(prefix="kaggle-vllm-m4-batch-ingest-"))
    try:
        with zipfile.ZipFile(batch_zip) as archive:
            archive.extractall(temporary)
        verified = verify_sha256_manifest(
            temporary, temporary / "BATCH_SHA256SUMS.txt"
        )
        if set(verified) != set(members) - {"BATCH_SHA256SUMS.txt"}:
            raise ResearchEvidenceError(
                "BATCH_SHA256SUMS.txt must cover every non-manifest member exactly"
            )
        bundled_runtime = temporary / "runtime.json"
        if sha256_file(bundled_runtime) != sha256_file(runtime_path):
            raise ResearchEvidenceError("separate runtime.json differs from batch bundle")
        manifest = load_object(temporary / "BATCH_MANIFEST.json")
        freeze_path, freeze = select_batch_source_freeze(
            repository, str(manifest.get("source_commit"))
        )
        frozen_notebook = repository / "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
        notebook_commit = str(freeze["notebook_pin_commit"])
        notebook_blob = _git_blob(
            repository,
            notebook_commit,
            "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb",
        )
        if notebook_blob is None:
            raise ResearchEvidenceError("cannot load frozen batch notebook from Git history")
        frozen_notebook = temporary / "FROZEN_BATCH_NOTEBOOK.ipynb"
        frozen_notebook.write_bytes(notebook_blob)
        source_equivalent = notebook_sources(
            notebook, allow_trailing_empty_code_cells=True
        ) == notebook_sources(frozen_notebook)
        if not source_equivalent and notebook_recovery_record is None:
            raise ResearchEvidenceError(
                "executed batch notebook source differs from frozen source"
            )
        if source_equivalent and notebook_recovery_record is not None:
            raise ResearchEvidenceError(
                "notebook recovery record is invalid for source-equivalent evidence"
            )
        recovery = None
        if notebook_recovery_record is not None:
            recovery = validate_reviewed_notebook_recovery(
                repository=repository,
                recovery_path=notebook_recovery_record,
                executed_notebook=notebook,
                frozen_notebook=frozen_notebook,
                batch_zip=batch_zip,
                runtime_path=runtime_path,
                manifest=manifest,
                freeze_path=freeze_path,
                freeze=freeze,
            )
        source_identity = load_object(temporary / "BATCH_SOURCE_IDENTITY.json")
        if source_identity.get("source_commit") != freeze["implementation_source_commit"]:
            raise ResearchEvidenceError("batch source identity differs from source freeze")
        if source_identity.get("batch_runner_sha256") != freeze["batch_runner_sha256"]:
            raise ResearchEvidenceError("batch runner identity differs from source freeze")
        expected_plan_path = freeze.get(
            "batch_plan_path", "research/M4_BATCH_EXECUTION_PLAN.json"
        )
        if source_identity.get("batch_plan_sha256") != freeze["batch_plan_sha256"]:
            raise ResearchEvidenceError("batch plan identity differs from source freeze")
        recorded_plan_path = source_identity.get(
            "batch_plan_path", "research/M4_BATCH_EXECUTION_PLAN.json"
        )
        if recorded_plan_path != expected_plan_path:
            raise ResearchEvidenceError("batch plan path differs from source freeze")
        notebook_source_identity_status = _validate_notebook_source_identity(
            source_identity, freeze, frozen_notebook
        )
        if "principal_queue_sha256" in freeze:
            if source_identity.get("principal_queue_sha256") != freeze[
                "principal_queue_sha256"
            ]:
                raise ResearchEvidenceError("principal queue identity differs from source freeze")
            if source_identity.get("principal_queue_path") != freeze.get(
                "principal_queue_path"
            ):
                raise ResearchEvidenceError("principal queue path differs from source freeze")
        plan = _load_frozen_object(
            repository,
            commit=freeze["implementation_source_commit"],
            relative=expected_plan_path,
            expected_sha256=freeze["batch_plan_sha256"],
        )
        maximum_batch_seconds = plan.get("resource_policy", {}).get(
            "maximum_batch_wall_clock_seconds"
        )
        if (
            isinstance(maximum_batch_seconds, bool)
            or not isinstance(maximum_batch_seconds, int)
            or maximum_batch_seconds <= 0
        ):
            raise ResearchEvidenceError("batch plan has invalid wall-clock limit")
        batch = select_batch(plan, str(manifest.get("batch_id")))
        allow_terminal_resource_continuation = terminal_resource_continuation_enabled(
            plan, batch
        )
        validate_batch_manifest(
            manifest,
            batch,
            expected_source_commit=freeze["implementation_source_commit"],
            allow_terminal_resource_continuation=allow_terminal_resource_continuation,
        )
        accepted = []
        invalid = []
        terminal_resource_shards = []
        outcomes = {item["shard_id"]: item for item in manifest["shards"]}
        planned = {
            item["shard_id"]: item for item in batch.get("execution_order", [])
        }
        for shard_id in manifest["actual_execution_order"]:
            outcome = outcomes[shard_id]
            historical_terminal_candidate = bool(
                allow_terminal_resource_continuation
                and outcome["status"] == "FAILED"
                and outcome.get("runner_returncode") == 3
                and outcome.get("post_shard_gpu_cleanup", {}).get("status") == "PASS"
            )
            if (
                outcome["status"] == "FAILED_RESOURCE_GATE"
                or historical_terminal_candidate
            ):
                model_matrix = _load_frozen_object(
                    repository,
                    commit=freeze["implementation_source_commit"],
                    relative="research/model_matrix.json",
                    expected_sha256=freeze["model_matrix_sha256"],
                )["models"]
                inner_zip = temporary / outcome["evidence_zip"]
                if sha256_file(inner_zip) != outcome["evidence_zip_sha256"]:
                    invalid.append(
                        {"shard_id": shard_id, "error": "inner ZIP digest mismatch"}
                    )
                    continue
                try:
                    terminal = _audit_terminal_resource_archive(
                        evidence_zip=inner_zip,
                        runtime=runtime,
                        expected_shard=planned[shard_id],
                        expected_model=model_matrix[planned[shard_id]["model_key"]],
                        outcome=outcome,
                        manifest=manifest,
                        expected_source_commit=freeze[
                            "implementation_source_commit"
                        ],
                        maximum_runtime_delay_seconds=(
                            MAX_RUNTIME_TO_EXECUTION_DELAY_SECONDS
                            + maximum_batch_seconds
                        ),
                    )
                    if historical_terminal_candidate:
                        terminal.update(
                            {
                                "historical_outer_outcome_status": "FAILED",
                                "historical_outer_reason": outcome.get("reason"),
                            }
                        )
                    terminal_resource_shards.append(terminal)
                except (ResearchEvidenceError, OSError, KeyError, ValueError) as error:
                    invalid.append({"shard_id": shard_id, "error": str(error)})
                continue
            if outcome["status"] != "COMPLETED":
                continue
            inner_zip = temporary / outcome["evidence_zip"]
            if sha256_file(inner_zip) != outcome["evidence_zip_sha256"]:
                invalid.append({"shard_id": shard_id, "error": "inner ZIP digest mismatch"})
                continue
            extracted: Path | None = None
            try:
                audit, extracted = audit_download(
                    repository=repository,
                    notebook=frozen_notebook if recovery is not None else notebook,
                    evidence_zip=inner_zip,
                    runtime_path=runtime_path,
                    expected_source_commit=freeze["implementation_source_commit"],
                    frozen_notebook=frozen_notebook,
                    allow_trailing_empty_notebook_cells=True,
                    maximum_runtime_delay_seconds=(
                        MAX_RUNTIME_TO_EXECUTION_DELAY_SECONDS + maximum_batch_seconds
                    ),
                )
                if audit["shard_id"] != shard_id:
                    raise ResearchEvidenceError("inner shard identity differs from manifest")
                shard_provenance = load_object(
                    extracted / "batch-shard-provenance.json"
                )
                _validate_batch_shard_provenance(
                    shard_provenance,
                    shard_id=shard_id,
                    outcome=outcome,
                    manifest=manifest,
                    expected_source_commit=freeze["implementation_source_commit"],
                )
                _stage_inner(
                    repository=repository,
                    extracted=extracted,
                    audit=audit,
                    notebook=notebook,
                    inner_zip=inner_zip,
                    runtime_path=runtime_path,
                    provenance={
                        "execution_mode": "batch_orchestrated",
                        "kaggle_session_id": manifest["session_id"],
                        "batch_id": manifest["batch_id"],
                        "within_session_order": outcome["within_session_order"],
                        "repetition": manifest["repetition"],
                        **(
                            {"notebook_provenance_recovery": recovery}
                            if recovery is not None
                            else {}
                        ),
                    },
                    recovery_record_path=notebook_recovery_record,
                )
                extracted = None
                accepted.append(audit)
            except (ResearchEvidenceError, OSError, ValueError) as error:
                if extracted is not None:
                    shutil.rmtree(extracted, ignore_errors=True)
                invalid.append({"shard_id": shard_id, "error": str(error)})
        verified_terminal_ids = {
            item["shard_id"] for item in terminal_resource_shards
        }
        failed_shards = [
            {
                "shard_id": item["shard_id"],
                "status": item["status"],
                "reason": item.get("reason"),
                "runner_returncode": item.get("runner_returncode"),
                "evidence_zip": item.get("evidence_zip"),
                "evidence_zip_sha256": item.get("evidence_zip_sha256"),
                "preservation_status": "PRESERVED_IN_ORIGINAL_OUTER_BATCH_FOR_REVIEW",
            }
            for item in manifest["shards"]
            if item["status"] == "FAILED"
            and item["shard_id"] not in verified_terminal_ids
        ]
        planned_not_executed = [
            item["shard_id"]
            for item in manifest["shards"]
            if item["status"] == "NOT_EXECUTED"
        ]
        review_required = bool(
            invalid
            or failed_shards
            or terminal_resource_shards
            or planned_not_executed
        )
        report = {
            "schema_version": OUTER_AUDIT_SCHEMA,
            "status": (
                "BATCH_REVIEW_REQUIRED"
                if review_required
                else "VERIFIED_BATCH_CANDIDATE"
            ),
            "batch_id": manifest["batch_id"],
            "session_id": manifest["session_id"],
            "batch_zip_sha256": sha256_file(batch_zip),
            "source_equivalent_notebook": source_equivalent,
            "notebook_source_identity_status": notebook_source_identity_status,
            "accepted_shards": accepted,
            "invalid_shards": invalid,
            "failed_shards": failed_shards,
            "planned_not_executed": planned_not_executed,
            "batch_manifest_status": manifest.get("status", "UNKNOWN"),
            "source_freeze": str(freeze_path.relative_to(repository)),
            "destination_status": "INDIVIDUAL_LOCAL_STAGING_ONLY_REQUIRES_REVIEWED_PROMOTION",
        }
        if recovery is not None:
            report["notebook_provenance_recovery"] = recovery
        if terminal_resource_shards:
            report["terminal_resource_shards"] = terminal_resource_shards
        return report
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
