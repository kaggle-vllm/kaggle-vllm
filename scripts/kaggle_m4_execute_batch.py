#!/usr/bin/env python3
"""Run multiple unchanged logical M4 shards in one provenance-tracked session."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_vllm.research.errors import ResearchEvidenceError
from kaggle_vllm.research.m4_batch import (
    BATCH_SCHEMA,
    check_disk_capacity,
    check_wall_clock,
    exact_hf_repo_cache_path,
    load_object,
    measure_shard_peaks,
    select_batch,
)
from kaggle_vllm.research.provenance import sha256_file, verify_sha256_manifest

BASE_RUNNER_PROJECTED_EVIDENCE_BYTES = 3_000_000_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_id(batch_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"m4-{batch_id}-{stamp}-{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{_utc_now()}] {message}\n")


def _git_identity(repository: Path) -> tuple[str, list[str]]:
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines()
    return commit, dirty


def _gpu_snapshot() -> dict[str, Any]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if gpu.returncode != 0 or compute.returncode != 0:
        raise ResearchEvidenceError("structured nvidia-smi query failed")
    devices = []
    for line in gpu.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ResearchEvidenceError("unexpected nvidia-smi GPU query format")
        devices.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "driver_version": fields[3],
                "memory_used_mib": float(fields[4]),
            }
        )
    compute_processes = []
    for line in compute.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        compute_processes.append({"pid": int(fields[0]), "used_memory_mib": float(fields[1])})
    return {"captured_at_utc": _utc_now(), "devices": devices, "compute_processes": compute_processes}


def _wait_for_gpu_cleanliness(
    *, baseline_pids: set[int], memory_limit_mib: float, timeout_seconds: int
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last = _gpu_snapshot()
    while True:
        new_pids = {
            item["pid"] for item in last["compute_processes"] if item["pid"] not in baseline_pids
        }
        memory_clean = all(
            device["memory_used_mib"] <= memory_limit_mib for device in last["devices"]
        )
        if not new_pids and memory_clean:
            return {"status": "PASS", "snapshot": last, "new_compute_pids": []}
        if time.monotonic() >= deadline:
            return {
                "status": "FAIL",
                "snapshot": last,
                "new_compute_pids": sorted(new_pids),
                "reason": "unexpected residual GPU process or memory; unrelated processes were not killed",
            }
        time.sleep(2)
        last = _gpu_snapshot()


def _make_zip(directory: Path, archive: Path) -> str:
    if archive.exists():
        raise ResearchEvidenceError(f"refusing to overwrite evidence ZIP: {archive}")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.is_symlink():
                raise ResearchEvidenceError(f"unexpected shard evidence entry: {path}")
            output.write(path, arcname=path.name)
    return sha256_file(archive)


def _add_batch_shard_provenance(
    directory: Path, provenance: dict[str, Any]
) -> None:
    runner_manifest = directory / "SHA256SUMS.txt"
    verify_sha256_manifest(directory, runner_manifest)
    shutil.copy2(runner_manifest, directory / "RUNNER_SHA256SUMS.txt")
    _write_json(directory / "batch-shard-provenance.json", provenance)
    members = sorted(
        path for path in directory.iterdir() if path.is_file() and path != runner_manifest
    )
    runner_manifest.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in members),
        encoding="utf-8",
    )
    verify_sha256_manifest(directory, runner_manifest)


def _cleanup_model_cache(
    *,
    hf_home: Path,
    hf_id: str,
    model_outcomes: list[dict[str, Any]],
    evidence_root: Path,
    bundle: Path,
) -> dict[str, Any]:
    for outcome in model_outcomes:
        if outcome["status"] != "COMPLETED":
            return {"status": "SKIPPED_INCOMPLETE_MODEL_GROUP"}
        archive = bundle / outcome["evidence_zip"]
        if sha256_file(archive) != outcome["evidence_zip_sha256"]:
            raise ResearchEvidenceError("refusing cache cleanup before verified ZIPs")
        if not (evidence_root / f"{outcome['shard_id']}-principal").is_dir():
            raise ResearchEvidenceError("refusing cache cleanup: evidence directory missing")
    cache = exact_hf_repo_cache_path(hf_home, hf_id)
    if not cache.exists():
        return {"status": "NOT_PRESENT", "exact_path": str(cache)}
    if cache.is_symlink() or not cache.is_dir():
        raise ResearchEvidenceError("exact model cache target is not a safe directory")
    shutil.rmtree(cache)
    for outcome in model_outcomes:
        if not (evidence_root / f"{outcome['shard_id']}-principal").is_dir():
            raise ResearchEvidenceError("evidence disappeared during model cache cleanup")
    return {"status": "DELETED_EXACT_MODEL_REPO_CACHE", "exact_path": str(cache)}


def _finalize_bundle(bundle: Path, manifest: dict[str, Any], outer_zip: Path) -> str:
    _write_json(bundle / "BATCH_MANIFEST.json", manifest)
    members = sorted(
        path for path in bundle.iterdir() if path.is_file() and path.name != "BATCH_SHA256SUMS.txt"
    )
    checksum = bundle / "BATCH_SHA256SUMS.txt"
    checksum.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in members),
        encoding="utf-8",
    )
    if outer_zip.exists():
        raise ResearchEvidenceError(f"refusing to overwrite outer batch ZIP: {outer_zip}")
    with zipfile.ZipFile(outer_zip, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted(bundle.iterdir(), key=lambda item: item.name):
            output.write(path, arcname=path.name)
    return sha256_file(outer_zip)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-identity", required=True)
    parser.add_argument("--batch-notebook-source-digest", required=True)
    parser.add_argument("--maximum-wall-clock-seconds", type=int)
    parser.add_argument("--minimum-remaining-seconds", type=int)
    parser.add_argument("--disk-safety-reserve-bytes", type=int)
    parser.add_argument("--gpu-cleanup-timeout-seconds", type=int, default=60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = args.repository.resolve()
    output_root = args.output_root.resolve()
    runtime_path = args.runtime.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite batch output: {output_root}")
    commit, dirty = _git_identity(repository)
    if commit != args.source_identity or dirty:
        raise SystemExit("batch source must be the exact clean reviewed commit")
    plan_path = repository / "research/M4_BATCH_EXECUTION_PLAN.json"
    plan = load_object(plan_path)
    batch = select_batch(plan, args.batch_id)
    policies = plan["resource_policy"]
    maximum_seconds = args.maximum_wall_clock_seconds or policies[
        "maximum_batch_wall_clock_seconds"
    ]
    minimum_remaining = args.minimum_remaining_seconds or policies[
        "minimum_remaining_seconds_to_start_shard"
    ]
    reserve = args.disk_safety_reserve_bytes or policies["disk_safety_reserve_bytes"]
    runtime = load_object(runtime_path)
    model_matrix = load_object(repository / "research/model_matrix.json")["models"]
    started_monotonic = time.monotonic()
    started_utc = _utc_now()
    session_id = _session_id(args.batch_id)
    evidence_root = output_root / "evidence"
    bundle = output_root / "bundle"
    evidence_root.mkdir(parents=True)
    bundle.mkdir()
    batch_log = bundle / "batch.log"
    shutil.copy2(runtime_path, bundle / "runtime.json")
    baseline = _gpu_snapshot()
    baseline_pids = {item["pid"] for item in baseline["compute_processes"]}
    source_identity = {
        "schema_version": "kaggle-vllm-m4-batch-source-identity-v1",
        "source_commit": commit,
        "source_dirty": False,
        "batch_runner_sha256": sha256_file(Path(__file__).resolve()),
        "base_shard_runner_sha256": sha256_file(
            repository / "scripts/kaggle_m4_multimodel_crossover.py"
        ),
        "batch_plan_sha256": sha256_file(plan_path),
        "batch_notebook_source_digest": args.batch_notebook_source_digest,
    }
    _write_json(bundle / "BATCH_SOURCE_IDENTITY.json", source_identity)
    outcomes = [
        {
            "shard_id": shard_id,
            "status": "SKIPPED_ALREADY_CANONICAL",
            "within_session_order": None,
        }
        for shard_id in batch["already_completed_skips"]
    ] + [
        {
            "shard_id": item["shard_id"],
            "model_key": item["model_key"],
            "workload": item["workload"],
            "repetition": item["repetition"],
            "within_session_order": item["within_session_order"],
            "status": "PLANNED",
        }
        for item in batch["execution_order"]
    ]
    manifest: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA,
        "status": "RUNNING",
        "execution_mode": "batch_orchestrated",
        "batch_id": args.batch_id,
        "session_id": session_id,
        "repetition": batch["repetition"],
        "start_utc": started_utc,
        "end_utc": None,
        "source_commit": commit,
        "runtime_manifest_sha256": sha256_file(runtime_path),
        "runtime_identity": runtime,
        "gpu_names": [device["name"] for device in baseline["devices"]],
        "gpu_uuids": [device["uuid"] for device in baseline["devices"]],
        "driver": [device["driver_version"] for device in baseline["devices"]],
        "cuda": runtime.get("environment", {}).get("torch_cuda"),
        "nccl": runtime.get("environment", {}).get("nccl"),
        "batch_runner_sha256": source_identity["batch_runner_sha256"],
        "batch_notebook_source_digest": args.batch_notebook_source_digest,
        "ordered_shard_ids": batch["ordered_shard_ids"],
        "already_completed_skips": batch["already_completed_skips"],
        "actual_execution_order": [],
        "shards": outcomes,
        "disk": {"samples": [], "high_water_used_bytes": 0, "safety_reserve_bytes": reserve},
        "wall_clock": {"maximum_seconds": maximum_seconds},
        "gpu_baseline": baseline,
        "cache_events": [],
        "stop_reason": None,
    }
    by_id = {item["shard_id"]: item for item in outcomes}
    _append_log(batch_log, f"session={session_id} batch={args.batch_id} source={commit}")
    return_code = 0
    stop = False
    current_model: str | None = None
    model_outcomes: list[dict[str, Any]] = []

    for index, item in enumerate(batch["execution_order"]):
        outcome = by_id[item["shard_id"]]
        if stop:
            outcome["status"] = "NOT_EXECUTED"
            outcome["reason"] = "NOT_EXECUTED_IN_THIS_BATCH_ATTEMPT"
            continue
        try:
            wall = check_wall_clock(
                time.monotonic() - started_monotonic,
                maximum_seconds=maximum_seconds,
                minimum_remaining_seconds=minimum_remaining,
            )
            model_key = item["model_key"]
            if current_model is not None and model_key != current_model:
                cache_event = _cleanup_model_cache(
                    hf_home=Path(os.environ["HF_HOME"]),
                    hf_id=model_matrix[current_model]["hf_id"],
                    model_outcomes=model_outcomes,
                    evidence_root=evidence_root,
                    bundle=bundle,
                )
                cache_event.update({"model_key": current_model, "at_utc": _utc_now()})
                manifest["cache_events"].append(cache_event)
                current_model = None
                model_outcomes = []
            cache_path = exact_hf_repo_cache_path(
                Path(os.environ["HF_HOME"]), model_matrix[model_key]["hf_id"]
            )
            selected_weight_bytes = int(model_matrix[model_key]["selected_weight_bytes"])
            projected_model = 0 if cache_path.is_dir() else selected_weight_bytes
            evidence_allowance = policies["projected_evidence_bytes_per_shard"]
            required_free = max(
                selected_weight_bytes + BASE_RUNNER_PROJECTED_EVIDENCE_BYTES,
                projected_model + evidence_allowance + reserve,
            )
            disk = check_disk_capacity(
                output_root.parent,
                projected_additional_bytes=required_free - reserve,
                reserve_bytes=reserve,
            )
            manifest["disk"]["samples"].append(
                {
                    "before_shard": item["shard_id"],
                    "captured_at_utc": _utc_now(),
                    "model_cache_present": cache_path.is_dir(),
                    "selected_weight_bytes": selected_weight_bytes,
                    "base_runner_required_free_bytes": selected_weight_bytes
                    + BASE_RUNNER_PROJECTED_EVIDENCE_BYTES,
                    **disk,
                }
            )
            manifest["disk"]["high_water_used_bytes"] = max(
                manifest["disk"]["high_water_used_bytes"], disk["used_bytes"]
            )
            current_model = model_key
            outcome.update({"status": "RUNNING", "start_utc": _utc_now(), "wall_clock_guard": wall})
            manifest["actual_execution_order"].append(item["shard_id"])
            command = [
                sys.executable,
                str(repository / "scripts/kaggle_m4_multimodel_crossover.py"),
                "--repository",
                str(repository),
                "--output-root",
                str(evidence_root),
                "--model-key",
                model_key,
                "--workload",
                item["workload"],
                "--repetition",
                str(item["repetition"]),
                "--mode",
                "principal",
                "--source-identity",
                commit,
            ]
            _append_log(batch_log, f"START {item['shard_id']} command={json.dumps(command)}")
            completed = subprocess.run(
                command,
                cwd=repository,
                env=dict(os.environ),
                check=False,
                capture_output=True,
                text=True,
            )
            _append_log(batch_log, f"STDOUT {item['shard_id']}\n{completed.stdout or '<empty>'}")
            _append_log(batch_log, f"STDERR {item['shard_id']}\n{completed.stderr or '<empty>'}")
            directory = evidence_root / f"{item['shard_id']}-principal"
            if not directory.is_dir():
                raise ResearchEvidenceError(
                    f"runner returned {completed.returncode} without evidence directory"
                )
            verify_sha256_manifest(directory, directory / "SHA256SUMS.txt")
            shard_end_utc = _utc_now()
            outcome.update(
                {
                    "end_utc": shard_end_utc,
                    "runner_returncode": completed.returncode,
                }
            )
            _add_batch_shard_provenance(
                directory,
                {
                    "schema_version": "kaggle-vllm-m4-batch-shard-provenance-v1",
                    "execution_mode": "batch_orchestrated",
                    "batch_id": args.batch_id,
                    "session_id": session_id,
                    "repetition": item["repetition"],
                    "within_session_order": item["within_session_order"],
                    "shard_id": item["shard_id"],
                    "start_utc": outcome["start_utc"],
                    "end_utc": shard_end_utc,
                    "source_commit": commit,
                    "runner_returncode": completed.returncode,
                },
            )
            archive = bundle / f"{item['shard_id']}-principal.zip"
            archive_sha = _make_zip(directory, archive)
            outcome.update(
                {
                    "evidence_directory": str(directory),
                    "evidence_zip": archive.name,
                    "evidence_zip_sha256": archive_sha,
                }
            )
            peaks = measure_shard_peaks(directory)
            outcome.update(
                {
                    "resource_guard": peaks,
                }
            )
            model_outcomes.append(outcome)
            if completed.returncode != 0:
                outcome["status"] = "FAILED"
                outcome["reason"] = "PRINCIPAL_RUNNER_NONZERO_RETURN_CODE"
                stop = True
                return_code = 2
            else:
                outcome["status"] = "COMPLETED"
            cleanup = _wait_for_gpu_cleanliness(
                baseline_pids=baseline_pids,
                memory_limit_mib=policies["idle_gpu_memory_limit_mib_per_gpu"],
                timeout_seconds=args.gpu_cleanup_timeout_seconds,
            )
            outcome["post_shard_gpu_cleanup"] = cleanup
            if cleanup["status"] != "PASS":
                stop = True
                return_code = 2
                manifest["stop_reason"] = "GPU_CLEANLINESS_GUARD"
            _append_log(batch_log, f"END {item['shard_id']} status={outcome['status']}")
        except (ResearchEvidenceError, OSError, KeyError, ValueError) as error:
            outcome.update({"status": "FAILED" if outcome["status"] == "RUNNING" else "NOT_EXECUTED", "end_utc": _utc_now(), "reason": str(error)})
            if outcome["status"] == "NOT_EXECUTED":
                outcome["reason"] = f"NOT_EXECUTED_IN_THIS_BATCH_ATTEMPT: {error}"
                if manifest["actual_execution_order"] and manifest["actual_execution_order"][-1] == item["shard_id"]:
                    manifest["actual_execution_order"].pop()
            manifest["stop_reason"] = str(error)
            _append_log(batch_log, f"STOP {item['shard_id']} reason={error}")
            stop = True
            return_code = 2

    if current_model is not None and model_outcomes:
        try:
            cache_event = _cleanup_model_cache(
                hf_home=Path(os.environ["HF_HOME"]),
                hf_id=model_matrix[current_model]["hf_id"],
                model_outcomes=model_outcomes,
                evidence_root=evidence_root,
                bundle=bundle,
            )
            cache_event.update({"model_key": current_model, "at_utc": _utc_now()})
            manifest["cache_events"].append(cache_event)
        except ResearchEvidenceError as error:
            manifest["cache_events"].append(
                {"model_key": current_model, "status": "FAILED_SAFE_NO_DELETE", "reason": str(error)}
            )
            manifest["stop_reason"] = manifest["stop_reason"] or str(error)
            return_code = 2
    for outcome in outcomes:
        if outcome["status"] == "PLANNED":
            outcome["status"] = "NOT_EXECUTED"
            outcome["reason"] = "NOT_EXECUTED_IN_THIS_BATCH_ATTEMPT"
    manifest["end_utc"] = _utc_now()
    manifest["batch_wall_clock_seconds"] = time.monotonic() - started_monotonic
    completed_count = sum(item["status"] == "COMPLETED" for item in outcomes)
    failed_count = sum(item["status"] == "FAILED" for item in outcomes)
    not_executed_count = sum(item["status"] == "NOT_EXECUTED" for item in outcomes)
    manifest["status"] = (
        "COMPLETED"
        if not failed_count and not not_executed_count
        else "PARTIAL_COMPLETION"
        if not failed_count and completed_count
        else "STOPPED_ON_FAILURE"
    )
    manifest["counts"] = {
        "completed": completed_count,
        "failed": failed_count,
        "not_executed": not_executed_count,
        "skipped_already_canonical": len(batch["already_completed_skips"]),
    }
    outer_zip = output_root.parent / f"m4-batch-{args.batch_id}.zip"
    outer_sha = _finalize_bundle(bundle, manifest, outer_zip)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "batch_id": args.batch_id,
                "session_id": session_id,
                "outer_zip": str(outer_zip),
                "outer_zip_sha256": outer_sha,
                "completed": completed_count,
                "remaining": not_executed_count,
            },
            indent=2,
        )
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
