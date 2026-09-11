from __future__ import annotations

import hashlib
import json
import shutil
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggle_vllm.research import m4_batch
from kaggle_vllm.research.errors import ResearchEvidenceError
from kaggle_vllm.research.m4_batch import (
    check_disk_capacity,
    check_wall_clock,
    inspect_batch_zip,
    measure_shard_peaks,
    select_batch,
    stage_batch_download,
    validate_batch_manifest,
    verify_batch_source_freeze,
)
from kaggle_vllm.research.m4_ingest import notebook_sources
from kaggle_vllm.research.provenance import sha256_file, verify_sha256_manifest
from scripts.kaggle_m4_execute_batch import (
    _add_batch_shard_provenance,
    _cleanup_model_cache,
)

ROOT = Path(__file__).resolve().parents[1]


def _batch() -> dict:
    plan = json.loads((ROOT / "research/M4_BATCH_EXECUTION_PLAN.json").read_text())
    return select_batch(plan, "fill-r00")


def _manifest(batch: dict, *, completed: int = 0) -> dict:
    session = "m4-fill-r00-20260911T010203Z-1234abcd"
    shards = [
        {
            "shard_id": shard_id,
            "status": "SKIPPED_ALREADY_CANONICAL",
            "within_session_order": None,
        }
        for shard_id in batch["already_completed_skips"]
    ]
    actual = []
    for index, shard_id in enumerate(batch["ordered_shard_ids"], 1):
        outcome = {
            "shard_id": shard_id,
            "within_session_order": index,
            "status": "NOT_EXECUTED",
        }
        if index <= completed:
            outcome.update(
                {
                    "status": "COMPLETED",
                    "evidence_zip": f"{shard_id}-principal.zip",
                    "evidence_zip_sha256": "0" * 64,
                }
            )
            actual.append(shard_id)
        shards.append(outcome)
    return {
        "schema_version": m4_batch.BATCH_SCHEMA,
        "batch_id": batch["batch_id"],
        "session_id": session,
        "repetition": batch["repetition"],
        "execution_mode": "batch_orchestrated",
        "source_commit": "a" * 40,
        "ordered_shard_ids": batch["ordered_shard_ids"],
        "already_completed_skips": batch["already_completed_skips"],
        "actual_execution_order": actual,
        "shards": shards,
    }


def test_select_batch_rejects_mixed_repetition() -> None:
    plan = {
        "schema_version": m4_batch.BATCH_PLAN_SCHEMA,
        "batches": [
            {
                "batch_id": "bad",
                "repetition": 0,
                "execution_order": [
                    {"model_key": "m", "workload": "w", "repetition": 1}
                ],
            }
        ],
    }
    with pytest.raises(ResearchEvidenceError, match="mixes repetition"):
        select_batch(plan, "bad")


def test_disk_and_wall_clock_guards() -> None:
    usage = lambda _path: SimpleNamespace(total=1000, used=200, free=800)
    report = check_disk_capacity(
        Path("/unused"),
        projected_additional_bytes=500,
        reserve_bytes=300,
        usage=usage,
    )
    assert report["required_free_bytes"] == 800
    with pytest.raises(ResearchEvidenceError, match="insufficient disk"):
        check_disk_capacity(
            Path("/unused"),
            projected_additional_bytes=501,
            reserve_bytes=300,
            usage=usage,
        )
    assert check_wall_clock(
        100, maximum_seconds=1000, minimum_remaining_seconds=900
    )["remaining_seconds"] == 900
    with pytest.raises(ResearchEvidenceError, match="wall-clock"):
        check_wall_clock(101, maximum_seconds=1000, minimum_remaining_seconds=900)


@pytest.mark.parametrize("peak,accepted", [(14848.0, True), (14849.0, False)])
def test_vram_guard_is_independent_per_gpu(
    tmp_path: Path, peak: float, accepted: bool
) -> None:
    rows = [
        {"gpu_index": 0, "memory_used_mib": 12000.0},
        {"gpu_index": 1, "memory_used_mib": peak},
    ]
    telemetry = [
        {"index": 0, "memory_used_mib": 12000.0},
        {"index": 1, "memory_used_mib": peak},
    ]
    (tmp_path / "cell.resources.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    (tmp_path / "cell.telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry)
    )
    if accepted:
        assert measure_shard_peaks(tmp_path)["gpu1_maximum_observed_mib"] == peak
    else:
        with pytest.raises(ResearchEvidenceError, match="GPU1"):
            measure_shard_peaks(tmp_path)


def test_outer_zip_rejects_traversal_and_duplicate(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape", "bad")
    with pytest.raises(ResearchEvidenceError, match="unsafe batch ZIP path"):
        inspect_batch_zip(traversal)

    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("runtime.json", "{}")
            archive.writestr("runtime.json", "{}")
    with pytest.raises(ResearchEvidenceError, match="duplicate batch ZIP member"):
        inspect_batch_zip(duplicate)


def test_batch_notebook_allows_only_inert_trailing_empty_code_cells(
    tmp_path: Path,
) -> None:
    frozen = tmp_path / "frozen.ipynb"
    executed = tmp_path / "executed.ipynb"
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "id": "source",
                "metadata": {},
                "source": ["print('frozen')\n"],
                "outputs": [],
                "execution_count": None,
            }
        ],
    }
    frozen.write_text(json.dumps(notebook))
    notebook["cells"].append(
        {
            "cell_type": "code",
            "id": "kaggle-added-empty-cell",
            "metadata": {},
            "source": [],
            "outputs": [],
            "execution_count": None,
        }
    )
    executed.write_text(json.dumps(notebook))
    assert notebook_sources(executed) != notebook_sources(frozen)
    assert notebook_sources(
        executed, allow_trailing_empty_code_cells=True
    ) == notebook_sources(frozen)

    notebook["cells"][-1]["source"] = ["print('drift')\n"]
    executed.write_text(json.dumps(notebook))
    assert notebook_sources(
        executed, allow_trailing_empty_code_cells=True
    ) != notebook_sources(frozen)


def test_partial_manifest_and_duplicate_shards() -> None:
    batch = _batch()
    manifest = _manifest(batch, completed=2)
    validate_batch_manifest(manifest, batch, expected_source_commit="a" * 40)
    manifest["shards"][-1]["shard_id"] = manifest["shards"][-2]["shard_id"]
    with pytest.raises(ResearchEvidenceError, match="duplicate shard IDs"):
        validate_batch_manifest(manifest, batch, expected_source_commit="a" * 40)


def test_manifest_enforces_fail_fast_execution_prefix() -> None:
    batch = _batch()
    manifest = _manifest(batch, completed=3)
    manifest["shards"][2]["status"] = "FAILED"
    with pytest.raises(ResearchEvidenceError, match="not a prefix"):
        validate_batch_manifest(manifest, batch, expected_source_commit="a" * 40)


def test_v2_manifest_binds_protocol_revision() -> None:
    batch = _batch()
    batch["protocol_amendment_version"] = "M4-BATCH-2"
    manifest = _manifest(batch)
    manifest["protocol_amendment_version"] = "M4-BATCH-2"
    validate_batch_manifest(manifest, batch, expected_source_commit="a" * 40)
    manifest["protocol_amendment_version"] = "M4-BATCH-1"
    with pytest.raises(ResearchEvidenceError, match="protocol amendment"):
        validate_batch_manifest(manifest, batch, expected_source_commit="a" * 40)


def test_batch_source_freeze_validates_and_rejects_drift(tmp_path: Path) -> None:
    assert verify_batch_source_freeze(ROOT)["package_version"] == "0.2.0"
    assert verify_batch_source_freeze(
        ROOT, ROOT / "research/M4_BATCH_SOURCE_FREEZE_V2.json"
    )["protocol_amendment_version"] == "M4-BATCH-2"
    repository = tmp_path / "repository"
    paths = [
        "research/M4_BATCH_SOURCE_FREEZE.json",
        "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb",
        "scripts/kaggle_m4_execute_batch.py",
        "scripts/kaggle_m4_multimodel_crossover.py",
        "research/M4_EXECUTION_PLAN.json",
        "research/model_matrix.json",
        "research/m4_protocol.json",
        "research/M4_BATCH_EXECUTION_PLAN.json",
        "research/M4_BATCH_PROTOCOL_AMENDMENT.md",
    ]
    for relative in paths:
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    (repository / "scripts/kaggle_m4_execute_batch.py").write_text("drift")
    with pytest.raises(ResearchEvidenceError, match="source-freeze mismatch"):
        verify_batch_source_freeze(repository)


def test_batch_sidecar_preserves_runner_manifest_and_is_hashed(tmp_path: Path) -> None:
    evidence = tmp_path / "shard"
    evidence.mkdir()
    payload = evidence / "payload.json"
    payload.write_text("{}")
    (evidence / "SHA256SUMS.txt").write_text(
        f"{sha256_file(payload)}  payload.json\n"
    )
    _add_batch_shard_provenance(
        evidence,
        {
            "schema_version": "kaggle-vllm-m4-batch-shard-provenance-v1",
            "session_id": "fixture",
        },
    )
    verified = verify_sha256_manifest(evidence, evidence / "SHA256SUMS.txt")
    assert set(verified) == {
        "RUNNER_SHA256SUMS.txt",
        "batch-shard-provenance.json",
        "payload.json",
    }


def test_cache_cleanup_deletes_only_exact_completed_model_cache(tmp_path: Path) -> None:
    hf_home = tmp_path / "hf"
    target = hf_home / "hub/models--Qwen--Qwen2.5-3B-Instruct"
    sibling = hf_home / "hub/models--other--keep"
    target.mkdir(parents=True)
    sibling.mkdir()
    (target / "weights").write_text("fixture")
    evidence_root = tmp_path / "evidence"
    directory = evidence_root / "qwen25_3b-short-r02-principal"
    directory.mkdir(parents=True)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    archive = bundle / "qwen25_3b-short-r02-principal.zip"
    archive.write_bytes(b"fixture archive")
    outcome = {
        "shard_id": "qwen25_3b-short-r02",
        "status": "COMPLETED",
        "evidence_zip": archive.name,
        "evidence_zip_sha256": sha256_file(archive),
    }
    result = _cleanup_model_cache(
        hf_home=hf_home,
        hf_id="Qwen/Qwen2.5-3B-Instruct",
        model_outcomes=[outcome],
        evidence_root=evidence_root,
        bundle=bundle,
    )
    assert result["status"] == "DELETED_EXACT_MODEL_REPO_CACHE"
    assert not target.exists()
    assert sibling.is_dir()
    assert directory.is_dir()


def _write_outer(root: Path, manifest: dict, inner: dict[str, bytes]) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    files = {
        "BATCH_MANIFEST.json": json.dumps(manifest).encode(),
        "BATCH_SOURCE_IDENTITY.json": json.dumps(
            {
                "source_commit": "a" * 40,
                "batch_runner_sha256": "b" * 64,
                "batch_plan_sha256": sha256_file(
                    ROOT / "research/M4_BATCH_EXECUTION_PLAN.json"
                ),
                "batch_notebook_source_digest": "c" * 64,
            }
        ).encode(),
        "batch.log": b"test fixture; no benchmark measurements\n",
        "runtime.json": b"{}",
        **inner,
    }
    for name, content in files.items():
        (bundle / name).write_bytes(content)
    checksum = "".join(
        f"{hashlib.sha256((bundle / name).read_bytes()).hexdigest()}  {name}\n"
        for name in sorted(files)
    )
    (bundle / "BATCH_SHA256SUMS.txt").write_text(checksum)
    outer = root / "outer.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        for path in sorted(bundle.iterdir()):
            archive.write(path, path.name)
    return outer


def test_batch_ingest_preserves_first_valid_inner_when_later_inner_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    (repository / "research").mkdir(parents=True)
    (repository / "kaggle-notebooks").mkdir()
    shutil.copy2(
        ROOT / "research/M4_BATCH_EXECUTION_PLAN.json",
        repository / "research/M4_BATCH_EXECUTION_PLAN.json",
    )
    notebook = tmp_path / "executed.ipynb"
    notebook.write_text("{}")
    (repository / "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb").write_text(
        "{}"
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    batch = _batch()
    manifest = _manifest(batch, completed=2)
    inner = {}
    for shard_id in manifest["actual_execution_order"]:
        name = f"{shard_id}-principal.zip"
        inner[name] = shard_id.encode()
        manifest["shards"][
            batch["ordered_shard_ids"].index(shard_id) + 1
        ]["evidence_zip_sha256"] = hashlib.sha256(inner[name]).hexdigest()
    outer = _write_outer(tmp_path, manifest, inner)
    monkeypatch.setattr(
        m4_batch,
        "select_batch_source_freeze",
        lambda _repository, _source: (
            repository / "research/M4_BATCH_SOURCE_FREEZE.json",
            {
                "implementation_source_commit": "a" * 40,
                "notebook_pin_commit": "b" * 40,
                "batch_runner_sha256": "b" * 64,
                "batch_plan_sha256": sha256_file(
                    ROOT / "research/M4_BATCH_EXECUTION_PLAN.json"
                ),
                "batch_notebook_source_digest": "c" * 64,
            },
        ),
    )
    monkeypatch.setattr(m4_batch, "_git_blob", lambda *_args: b"{}")
    monkeypatch.setattr(m4_batch, "notebook_sources", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(m4_batch, "verify_runtime", lambda _runtime: None)
    calls = 0

    def fake_audit(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ResearchEvidenceError("invalid second inner shard")
        extracted = Path(tmp_path / "audit-extracted")
        extracted.mkdir()
        (extracted / "evidence.txt").write_text("validated fixture")
        shard_id = manifest["actual_execution_order"][calls - 1]
        (extracted / "batch-shard-provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": "kaggle-vllm-m4-batch-shard-provenance-v1",
                    "execution_mode": "batch_orchestrated",
                    "shard_id": shard_id,
                    "batch_id": manifest["batch_id"],
                    "session_id": manifest["session_id"],
                    "repetition": manifest["repetition"],
                    "within_session_order": calls,
                    "source_commit": "a" * 40,
                }
            )
        )
        return (
            {
                "shard_id": shard_id,
                "evidence_zip_sha256": hashlib.sha256(
                    kwargs["evidence_zip"].read_bytes()
                ).hexdigest(),
                "status": "VERIFIED_CANONICAL_CANDIDATE",
            },
            extracted,
        )

    monkeypatch.setattr(m4_batch, "audit_download", fake_audit)
    report = stage_batch_download(
        repository=repository,
        notebook=notebook,
        batch_zip=outer,
        runtime_path=runtime,
    )
    assert report["status"] == "BATCH_REVIEW_REQUIRED"
    assert len(report["accepted_shards"]) == 1
    assert len(report["invalid_shards"]) == 1
    staged = repository / ".local-evidence/m4-ingest"
    assert len(list(staged.iterdir())) == 1


def test_partial_batch_reports_failed_and_not_executed_without_promoting_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    (repository / "research").mkdir(parents=True)
    (repository / "kaggle-notebooks").mkdir()
    shutil.copy2(
        ROOT / "research/M4_BATCH_EXECUTION_PLAN.json",
        repository / "research/M4_BATCH_EXECUTION_PLAN.json",
    )
    notebook = tmp_path / "executed.ipynb"
    notebook.write_text("{}")
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    batch = _batch()
    manifest = _manifest(batch, completed=1)
    failed_id = batch["ordered_shard_ids"][1]
    manifest["actual_execution_order"].append(failed_id)
    manifest["shards"][2].update(
        {
            "status": "FAILED",
            "reason": "per-GPU VRAM limit exceeded",
            "runner_returncode": 3,
            "evidence_zip": f"{failed_id}-principal.zip",
            "evidence_zip_sha256": hashlib.sha256(failed_id.encode()).hexdigest(),
        }
    )
    manifest["status"] = "STOPPED_ON_FAILURE"
    completed_id = batch["ordered_shard_ids"][0]
    completed_name = f"{completed_id}-principal.zip"
    completed_bytes = completed_id.encode()
    manifest["shards"][1]["evidence_zip_sha256"] = hashlib.sha256(
        completed_bytes
    ).hexdigest()
    outer = _write_outer(
        tmp_path,
        manifest,
        {
            completed_name: completed_bytes,
            f"{failed_id}-principal.zip": failed_id.encode(),
        },
    )
    monkeypatch.setattr(
        m4_batch,
        "select_batch_source_freeze",
        lambda _repository, _source: (
            repository / "research/M4_BATCH_SOURCE_FREEZE.json",
            {
                "implementation_source_commit": "a" * 40,
                "notebook_pin_commit": "b" * 40,
                "batch_runner_sha256": "b" * 64,
                "batch_plan_sha256": sha256_file(
                    ROOT / "research/M4_BATCH_EXECUTION_PLAN.json"
                ),
                "batch_notebook_source_digest": "c" * 64,
            },
        ),
    )
    monkeypatch.setattr(m4_batch, "_git_blob", lambda *_args: b"{}")
    monkeypatch.setattr(m4_batch, "notebook_sources", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(m4_batch, "verify_runtime", lambda _runtime: None)

    def fake_audit(**_kwargs):
        extracted = tmp_path / "valid-inner"
        extracted.mkdir()
        (extracted / "batch-shard-provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": "kaggle-vllm-m4-batch-shard-provenance-v1",
                    "execution_mode": "batch_orchestrated",
                    "shard_id": completed_id,
                    "batch_id": manifest["batch_id"],
                    "session_id": manifest["session_id"],
                    "repetition": manifest["repetition"],
                    "within_session_order": 1,
                    "source_commit": "a" * 40,
                }
            )
        )
        return (
            {
                "shard_id": completed_id,
                "evidence_zip_sha256": hashlib.sha256(completed_bytes).hexdigest(),
                "status": "VERIFIED_CANONICAL_CANDIDATE",
            },
            extracted,
        )

    monkeypatch.setattr(m4_batch, "audit_download", fake_audit)
    report = stage_batch_download(
        repository=repository,
        notebook=notebook,
        batch_zip=outer,
        runtime_path=runtime,
    )
    assert report["status"] == "BATCH_REVIEW_REQUIRED"
    assert [item["shard_id"] for item in report["accepted_shards"]] == [
        completed_id
    ]
    assert [item["shard_id"] for item in report["failed_shards"]] == [failed_id]
    assert failed_id not in report["planned_not_executed"]
    assert len(report["planned_not_executed"]) == 9
