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
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ResearchEvidenceError
from .m4_ingest import (
    MAX_RUNTIME_TO_EXECUTION_DELAY_SECONDS,
    _check_existing_shard,
    audit_download,
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
    "NOT_EXECUTED",
    "SKIPPED_ALREADY_CANONICAL",
}
SESSION_ID = re.compile(r"^m4-[a-z0-9-]+-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
MAX_OUTER_MEMBERS = 128
MAX_OUTER_MEMBER_BYTES = 1024**3
MAX_OUTER_UNCOMPRESSED_BYTES = 4 * 1024**3


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


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchEvidenceError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ResearchEvidenceError(f"expected JSON object: {path}")
    return value


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


def exact_hf_repo_cache_path(hf_home: Path, hf_id: str) -> Path:
    parts = hf_id.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ResearchEvidenceError(f"unsafe Hugging Face repository ID: {hf_id}")
    hub = (hf_home / "hub").resolve()
    candidate = (hub / f"models--{parts[0]}--{parts[1]}").resolve()
    if candidate.parent != hub or not candidate.name.startswith("models--"):
        raise ResearchEvidenceError("cannot identify exact Hugging Face model cache path")
    return candidate


def measure_shard_peaks(evidence: Path) -> dict[str, Any]:
    peaks = {0: 0.0, 1: 0.0}
    sample_counts = {0: 0, 1: 0}
    ledgers = sorted(evidence.glob("*.resources.jsonl")) + sorted(
        evidence.glob("*.telemetry.jsonl")
    )
    if not ledgers:
        raise ResearchEvidenceError("shard has no resource or telemetry ledgers")
    for ledger in ledgers:
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
            sample_counts[gpu] += 1
            if memory > GPU_MEMORY_LIMIT_MIB:
                raise ResearchEvidenceError(
                    f"per-GPU VRAM limit exceeded: GPU{gpu} {memory} MiB > "
                    f"{GPU_MEMORY_LIMIT_MIB} MiB"
                )
    if any(count == 0 for count in sample_counts.values()):
        raise ResearchEvidenceError("resource evidence does not cover both physical GPUs")
    return {
        "gpu0_maximum_observed_mib": peaks[0],
        "gpu1_maximum_observed_mib": peaks[1],
        "per_gpu_limit_mib": GPU_MEMORY_LIMIT_MIB,
        "status": "PASS",
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
    manifest: dict[str, Any], batch: dict[str, Any], *, expected_source_commit: str
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
        if status_value in {"COMPLETED", "FAILED"}:
            if terminal_seen:
                raise ResearchEvidenceError("batch execution order is not a prefix")
            started.append(shard_id)
            if status_value == "FAILED":
                terminal_seen = True
        elif status_value == "NOT_EXECUTED":
            terminal_seen = True
        else:
            raise ResearchEvidenceError("planned batch row lacks a terminal outcome")
        if status_value == "COMPLETED" and (
            not outcome.get("evidence_zip") or not outcome.get("evidence_zip_sha256")
        ):
            raise ResearchEvidenceError("completed shard lacks ZIP identity")
    if manifest.get("actual_execution_order") != started:
        raise ResearchEvidenceError("actual execution order differs from shard outcomes")


def _stage_inner(
    *,
    repository: Path,
    extracted: Path,
    audit: dict[str, Any],
    notebook: Path,
    inner_zip: Path,
    runtime_path: Path,
    provenance: dict[str, Any],
) -> None:
    destination = repository / ".local-evidence/m4-ingest" / audit["evidence_zip_sha256"]
    _check_existing_shard(repository, audit["shard_id"], destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    originals = extracted / "original"
    originals.mkdir()
    for source in (notebook, inner_zip, runtime_path):
        shutil.copy2(source, originals / source.name)
    audit.update(provenance)
    audit["staged_at"] = str(destination)
    (extracted / "INGEST_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    shutil.move(str(extracted), destination)


def stage_batch_download(
    *, repository: Path, notebook: Path, batch_zip: Path, runtime_path: Path
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
        if notebook_sources(
            notebook, allow_trailing_empty_code_cells=True
        ) != notebook_sources(frozen_notebook):
            raise ResearchEvidenceError("executed batch notebook source differs from frozen source")
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
        if (
            source_identity.get("batch_notebook_source_digest")
            != freeze["batch_notebook_source_digest"]
        ):
            raise ResearchEvidenceError("batch notebook source identity differs from freeze")
        plan_path = repository / expected_plan_path
        if sha256_file(plan_path) != freeze["batch_plan_sha256"]:
            raise ResearchEvidenceError("current batch plan differs from source freeze")
        plan = load_object(plan_path)
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
        validate_batch_manifest(
            manifest,
            batch,
            expected_source_commit=freeze["implementation_source_commit"],
        )
        accepted = []
        invalid = []
        outcomes = {item["shard_id"]: item for item in manifest["shards"]}
        for shard_id in manifest["actual_execution_order"]:
            outcome = outcomes[shard_id]
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
                    notebook=notebook,
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
                provenance_checks = {
                    "schema": shard_provenance.get("schema_version")
                    == "kaggle-vllm-m4-batch-shard-provenance-v1",
                    "mode": shard_provenance.get("execution_mode")
                    == "batch_orchestrated",
                    "shard": shard_provenance.get("shard_id") == shard_id,
                    "batch": shard_provenance.get("batch_id")
                    == manifest["batch_id"],
                    "session": shard_provenance.get("session_id")
                    == manifest["session_id"],
                    "repetition": shard_provenance.get("repetition")
                    == manifest["repetition"],
                    "order": shard_provenance.get("within_session_order")
                    == outcome["within_session_order"],
                    "source": shard_provenance.get("source_commit")
                    == freeze["implementation_source_commit"],
                }
                if failed := [
                    name for name, passed in provenance_checks.items() if not passed
                ]:
                    raise ResearchEvidenceError(
                        f"inner batch-shard provenance mismatch: {failed}"
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
                    },
                )
                extracted = None
                accepted.append(audit)
            except (ResearchEvidenceError, OSError, ValueError) as error:
                if extracted is not None:
                    shutil.rmtree(extracted, ignore_errors=True)
                invalid.append({"shard_id": shard_id, "error": str(error)})
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
        ]
        planned_not_executed = [
            item["shard_id"]
            for item in manifest["shards"]
            if item["status"] == "NOT_EXECUTED"
        ]
        review_required = bool(invalid or failed_shards or planned_not_executed)
        return {
            "schema_version": OUTER_AUDIT_SCHEMA,
            "status": (
                "BATCH_REVIEW_REQUIRED"
                if review_required
                else "VERIFIED_BATCH_CANDIDATE"
            ),
            "batch_id": manifest["batch_id"],
            "session_id": manifest["session_id"],
            "batch_zip_sha256": sha256_file(batch_zip),
            "source_equivalent_notebook": True,
            "accepted_shards": accepted,
            "invalid_shards": invalid,
            "failed_shards": failed_shards,
            "planned_not_executed": planned_not_executed,
            "batch_manifest_status": manifest.get("status", "UNKNOWN"),
            "source_freeze": str(freeze_path.relative_to(repository)),
            "destination_status": "INDIVIDUAL_LOCAL_STAGING_ONLY_REQUIRES_REVIEWED_PROMOTION",
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
