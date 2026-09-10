import json
import zipfile
from pathlib import Path

import pytest

from kaggle_vllm.research import m4_ingest
from kaggle_vllm.research.errors import ResearchEvidenceError
from kaggle_vllm.research.m4_ingest import (
    SUCCESS_METRICS,
    _check_existing_shard,
    _validate_successful_cell,
    _verify_runtime_binding,
    inspect_zip,
    notebook_sources,
    verify_runtime,
)


def test_zip_audit_rejects_traversal(tmp_path: Path) -> None:
    candidate = tmp_path / "bad.zip"
    with zipfile.ZipFile(candidate, "w") as archive:
        archive.writestr("../escape", "bad")
    with pytest.raises(ResearchEvidenceError, match="unsafe ZIP path"):
        inspect_zip(candidate)


def test_zip_audit_rejects_noncanonical_and_oversized_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    noncanonical = tmp_path / "noncanonical.zip"
    with zipfile.ZipFile(noncanonical, "w") as archive:
        archive.writestr("./execution-start.json", "{}")
    with pytest.raises(ResearchEvidenceError, match="unsafe ZIP path"):
        inspect_zip(noncanonical)

    oversized = tmp_path / "oversized.zip"
    with zipfile.ZipFile(oversized, "w") as archive:
        archive.writestr("execution-start.json", "{}")
    monkeypatch.setattr(m4_ingest, "MAX_ZIP_MEMBER_BYTES", 1)
    with pytest.raises(ResearchEvidenceError, match="oversized ZIP member"):
        inspect_zip(oversized)


def test_notebook_source_comparison_ignores_outputs(tmp_path: Path) -> None:
    first = tmp_path / "first.ipynb"
    second = tmp_path / "second.ipynb"
    base = {"cells": [{"cell_type": "code", "id": "cell", "source": ["x = 1\n"], "outputs": []}]}
    first.write_text(json.dumps(base))
    base["cells"][0]["outputs"] = [{"output_type": "stream", "text": ["ok\n"]}]
    second.write_text(json.dumps(base))
    assert notebook_sources(first) == notebook_sources(second)


def test_runtime_gate_requires_two_sm75_t4s() -> None:
    runtime = {
        "profile": "kaggle-t4x2-cu128",
        "strict": True,
        "wheel": {
            "sha256": "5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c",
            "hf_repo_id": "waqasm86/kaggle-vllm-binaries",
            "hf_revision": "f6b4f10de54924ed6fe9e28cceab84eca7276ab6",
        },
        "environment": {
            "python": "3.12.13",
            "torch": "2.10.0+cu128",
            "torch_cuda": "12.8",
            "cuda_available": True,
            "nccl": "2.27.5",
            "driver_version": "580.159.04",
            "nvcc_version": "Cuda compilation tools, release 12.8, V12.8.93",
            "gpus": [
                {"name": "Tesla T4", "capability": [7, 5]},
                {"name": "Tesla T4", "capability": [7, 5]},
            ],
        },
    }
    verify_runtime(runtime)
    runtime["environment"]["gpus"][1]["capability"] = [8, 0]
    with pytest.raises(ResearchEvidenceError, match="runtime identity mismatch"):
        verify_runtime(runtime)


def test_runtime_manifest_is_temporally_bound_to_execution() -> None:
    runtime = {
        "completed_at": "2026-09-11T10:00:00+00:00",
        "environment": {"python": "3.12.13", "torch": "2.10.0+cu128"},
    }
    start = {
        "started_at_utc": "2026-09-11T10:01:00+00:00",
        "runtime": [
            {},
            {
                "python": "3.12.13",
                "torch": "2.10.0+cu128",
                "gpu_count": 2,
            },
        ],
    }
    _verify_runtime_binding(runtime, start)
    start["started_at_utc"] = "2026-09-11T12:00:00+00:00"
    with pytest.raises(ResearchEvidenceError, match="temporally bound"):
        _verify_runtime_binding(runtime, start)


def _successful_cell(tmp_path: Path) -> tuple[dict, dict, str]:
    stem = "qwen25_3b-short-r00-tp1-c01"
    requests = [
        {
            "request_id": f"request-{index:04d}",
            "status": "completed",
            "concurrency": 1,
            "input_tokens": 128,
            "output_tokens": 64,
        }
        for index in range(20)
    ]
    (tmp_path / f"{stem}-requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8"
    )
    resources = [
        {
            "gpu_index": index,
            "phase": stem,
            "memory_used_mib": 12_000.0 if index == 0 else 3.0,
            "system_used_bytes": 10_000_000_000,
        }
        for index in (0, 1)
    ]
    (tmp_path / f"{stem}.resources.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in resources), encoding="utf-8"
    )
    telemetry = [
        {"index": index, "memory_used_mib": 12_000.0 if index == 0 else 3.0}
        for index in (0, 1)
    ]
    (tmp_path / f"{stem}.telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry), encoding="utf-8"
    )
    prompt_hash = "a" * 64
    result = {
        "status": "executed",
        "engine": {"enable_prefix_caching": False},
        "workload": {
            "concurrency": 1,
            "warmup_requests": 1,
            "total_requests": 20,
            "max_output_tokens": 64,
            "ignore_eos": True,
            "prompt_manifest_sha256": prompt_hash,
        },
        "measurements": {
            "successful_requests": 20,
            "failed_requests": 0,
            "input_tokens": 20 * 128,
            "output_tokens": 20 * 64,
            "input_tokens_per_request": {"count": 20, "min": 128, "max": 128},
            "output_tokens_per_request": {"count": 20, "min": 64, "max": 64},
            "requests": {
                "file": f"{stem}-requests.jsonl",
                "count": 20,
                "storage": "jsonl",
            },
        },
        "server": {"visible_physical_gpu_indices": [0]},
        "gpu_telemetry": {
            "telemetry_sample_count": 2,
            "summaries": [
                {"index": 0, "peak_memory_used_mib": 12_000.0},
                {"index": 1, "peak_memory_used_mib": 3.0},
            ],
        },
    }
    row = {
        "measured_requests": 20,
        "warmup_requests": 1,
        "request_failures": 0,
        "oom": False,
        **dict.fromkeys(SUCCESS_METRICS, 1.0),
        "maximum_vram_mib": 12_000.0,
        "maximum_system_ram_bytes": 10_000_000_000,
    }
    return result, row, stem


def test_successful_cell_validates_request_tokens_and_resources(tmp_path: Path) -> None:
    result, row, stem = _successful_cell(tmp_path)
    _validate_successful_cell(
        root=tmp_path,
        result=result,
        row=row,
        stem=stem,
        concurrency=1,
        input_tokens=128,
        output_tokens=64,
        prompt_sha256="a" * 64,
    )
    result["measurements"]["successful_requests"] = 19
    with pytest.raises(ResearchEvidenceError, match="successful-cell invariant"):
        _validate_successful_cell(
            root=tmp_path,
            result=result,
            row=row,
            stem=stem,
            concurrency=1,
            input_tokens=128,
            output_tokens=64,
            prompt_sha256="a" * 64,
        )


def test_successful_cell_rejects_per_gpu_vram_limit(tmp_path: Path) -> None:
    result, row, stem = _successful_cell(tmp_path)
    resource_path = tmp_path / f"{stem}.resources.jsonl"
    rows = [json.loads(line) for line in resource_path.read_text().splitlines()]
    rows[1]["memory_used_mib"] = 14_849.0
    resource_path.write_text(
        "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
    )
    with pytest.raises(ResearchEvidenceError, match="per-GPU VRAM limit exceeded"):
        _validate_successful_cell(
            root=tmp_path,
            result=result,
            row=row,
            stem=stem,
            concurrency=1,
            input_tokens=128,
            output_tokens=64,
            prompt_sha256="a" * 64,
        )


def test_successful_cell_accepts_exact_14848_mib_per_gpu(tmp_path: Path) -> None:
    result, row, stem = _successful_cell(tmp_path)
    resource_path = tmp_path / f"{stem}.resources.jsonl"
    rows = [json.loads(line) for line in resource_path.read_text().splitlines()]
    rows[0]["memory_used_mib"] = 14_848.0
    resource_path.write_text(
        "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
    )
    telemetry_path = tmp_path / f"{stem}.telemetry.jsonl"
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    telemetry[0]["memory_used_mib"] = 14_848.0
    telemetry_path.write_text(
        "".join(json.dumps(item) + "\n" for item in telemetry), encoding="utf-8"
    )
    result["gpu_telemetry"]["summaries"][0]["peak_memory_used_mib"] = 14_848.0
    row["maximum_vram_mib"] = 14_848.0
    _validate_successful_cell(
        root=tmp_path,
        result=result,
        row=row,
        stem=stem,
        concurrency=1,
        input_tokens=128,
        output_tokens=64,
        prompt_sha256="a" * 64,
    )


def test_successful_cell_rejects_telemetry_per_gpu_vram_limit(tmp_path: Path) -> None:
    result, row, stem = _successful_cell(tmp_path)
    telemetry_path = tmp_path / f"{stem}.telemetry.jsonl"
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    telemetry[1]["memory_used_mib"] = 14_849.0
    telemetry_path.write_text(
        "".join(json.dumps(item) + "\n" for item in telemetry), encoding="utf-8"
    )
    with pytest.raises(ResearchEvidenceError, match="per-GPU VRAM limit exceeded"):
        _validate_successful_cell(
            root=tmp_path,
            result=result,
            row=row,
            stem=stem,
            concurrency=1,
            input_tokens=128,
            output_tokens=64,
            prompt_sha256="a" * 64,
        )


def test_staging_rejects_second_candidate_for_same_shard(tmp_path: Path) -> None:
    staging = tmp_path / ".local-evidence/m4-ingest/existing-hash"
    staging.mkdir(parents=True)
    (staging / "INGEST_AUDIT.json").write_text(
        json.dumps({"shard_id": "qwen25_3b-short-r00"}), encoding="utf-8"
    )
    with pytest.raises(ResearchEvidenceError, match="already staged shard"):
        _check_existing_shard(
            tmp_path,
            "qwen25_3b-short-r00",
            tmp_path / ".local-evidence/m4-ingest/different-hash",
        )
