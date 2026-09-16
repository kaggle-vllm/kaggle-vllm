from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggle_vllm.research import m4_batch
from kaggle_vllm.research.errors import ResearchEvidenceError
from kaggle_vllm.research.m4_batch import (
    COMPLETED_WITH_TERMINAL_OUTCOMES,
    check_disk_capacity,
    check_wall_clock,
    inspect_batch_zip,
    measure_shard_peaks,
    select_batch,
    stage_batch_download,
    validate_batch_against_queue,
    validate_batch_manifest,
    validate_current_notebook_self_digest,
    verify_batch_source_freeze,
    verify_terminal_resource_gate,
)
from kaggle_vllm.research.m4_ingest import notebook_sources
from kaggle_vllm.research.provenance import sha256_file, verify_sha256_manifest
from scripts import kaggle_m4_execute_batch as batch_runner
from scripts.generate_m4_batch_source_freeze_v4 import (
    build_freeze_v10,
    build_freeze_v11,
    build_freeze_v12,
)
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


def test_current_batch_passes_authoritative_no_rerun_queue() -> None:
    plan = json.loads((ROOT / "research/M4_REMAINING_EXECUTION_PLAN.json").read_text())
    queue = json.loads((ROOT / "research/M4_PRINCIPAL_EXECUTION_QUEUE.json").read_text())
    batch = select_batch(plan, "r02-phi")
    result = validate_batch_against_queue(batch, queue)
    assert result["status"] == "PASS_NO_SETTLED_SHARD_RESCHEDULED"
    assert result["queued_shard_ids"] == [
        "phi4_mini-prefill_heavy-r02",
        "phi4_mini-short-r02",
        "phi4_mini-balanced-r02",
    ]
    assert batch["review_required_exclusions"] == []
    assert batch["logical_shard_count"] == 3
    assert batch["serving_cell_count"] == 36


def test_promoted_canonical_batch_cannot_be_rescheduled() -> None:
    queue = json.loads((ROOT / "research/M4_PRINCIPAL_EXECUTION_QUEUE.json").read_text())
    remaining = json.loads(
        (ROOT / "research/M4_REMAINING_EXECUTION_PLAN.json").read_text()
    )
    with pytest.raises(ResearchEvidenceError, match="unknown or duplicate"):
        select_batch(remaining, "fill-r01-qwen")
    with pytest.raises(ResearchEvidenceError, match="unknown or duplicate"):
        select_batch(remaining, "r02-llama")
    with pytest.raises(ResearchEvidenceError, match="unknown or duplicate"):
        select_batch(remaining, "r02-ministral")
    with pytest.raises(ResearchEvidenceError, match="unknown or duplicate"):
        select_batch(remaining, "r02-qwen")
    stale_batch = {
        "ordered_shard_ids": [
            "qwen25_3b-balanced-r01",
            "qwen25_3b-prefill_heavy-r01",
        ],
        "execution_order": [
            {"shard_id": "qwen25_3b-balanced-r01"},
            {"shard_id": "qwen25_3b-prefill_heavy-r01"},
        ],
        "already_completed_skips": [],
        "review_required_exclusions": [],
    }
    with pytest.raises(ResearchEvidenceError, match="refusing to reschedule settled"):
        validate_batch_against_queue(stale_batch, queue)
    stale_llama = {
        "ordered_shard_ids": [
            "llama32_3b-prefill_heavy-r02",
            "llama32_3b-short-r02",
            "llama32_3b-balanced-r02",
        ],
        "execution_order": [
            {"shard_id": "llama32_3b-prefill_heavy-r02"},
            {"shard_id": "llama32_3b-short-r02"},
            {"shard_id": "llama32_3b-balanced-r02"},
        ],
        "already_completed_skips": [],
        "review_required_exclusions": [],
    }
    with pytest.raises(ResearchEvidenceError, match="refusing to reschedule settled"):
        validate_batch_against_queue(stale_llama, queue)
    stale_qwen = {
        "ordered_shard_ids": [
            "qwen25_3b-prefill_heavy-r02",
            "qwen25_3b-short-r02",
            "qwen25_3b-balanced-r02",
        ],
        "execution_order": [
            {"shard_id": "qwen25_3b-prefill_heavy-r02"},
            {"shard_id": "qwen25_3b-short-r02"},
            {"shard_id": "qwen25_3b-balanced-r02"},
        ],
        "already_completed_skips": [],
        "review_required_exclusions": [],
    }
    with pytest.raises(ResearchEvidenceError, match="refusing to reschedule settled"):
        validate_batch_against_queue(stale_qwen, queue)


def test_qwen_r02_attempts_remain_distinct_physical_sessions() -> None:
    evidence = json.loads((ROOT / "research/M4_EVIDENCE_STATUS.json").read_text())
    attempts = evidence["principal_batch_attempts"]
    first = attempts["r02-qwen-attempt-1"]
    second = attempts["r02-qwen-attempt-2"]
    assert first["session_id"] == "m4-r02-qwen-20260916T030342Z-ea0f6e11"
    assert second["session_id"] == "m4-r02-qwen-20260916T054736Z-95506297"
    assert first["session_id"] != second["session_id"]
    assert first["failed_resource_gate_shards"] == [
        "qwen25_3b-prefill_heavy-r02"
    ]
    assert second["completed_canonical_shards"] == [
        "qwen25_3b-short-r02",
        "qwen25_3b-balanced-r02",
    ]
    assert second["distinct_physical_session_from_attempt_1"] is True


def test_resource_gated_shard_cannot_be_rescheduled() -> None:
    queue = json.loads((ROOT / "research/M4_PRINCIPAL_EXECUTION_QUEUE.json").read_text())
    batch = {
        "ordered_shard_ids": ["qwen25_3b-prefill_heavy-r00"],
        "execution_order": [{"shard_id": "qwen25_3b-prefill_heavy-r00"}],
        "already_completed_skips": [],
        "review_required_exclusions": [],
    }
    with pytest.raises(ResearchEvidenceError, match="refusing to reschedule settled"):
        validate_batch_against_queue(batch, queue)
    batch["ordered_shard_ids"] = ["qwen25_3b-prefill_heavy-r02"]
    batch["execution_order"] = [{"shard_id": "qwen25_3b-prefill_heavy-r02"}]
    with pytest.raises(ResearchEvidenceError, match="refusing to reschedule settled"):
        validate_batch_against_queue(batch, queue)


def test_zero_remaining_batch_refuses_execution_cleanly() -> None:
    queue = json.loads((ROOT / "research/M4_PRINCIPAL_EXECUTION_QUEUE.json").read_text())
    batch = {
        "ordered_shard_ids": [],
        "execution_order": [],
        "already_completed_skips": ["qwen25_3b-short-r00"],
        "review_required_exclusions": [],
    }
    with pytest.raises(ResearchEvidenceError, match="zero remaining shards"):
        validate_batch_against_queue(batch, queue)


def test_pr_prose_cannot_override_machine_readable_queue() -> None:
    queue = json.loads((ROOT / "research/M4_PRINCIPAL_EXECUTION_QUEUE.json").read_text())
    stale_pr_claim = "Next operational batch: fill-r01-ministral"
    assert stale_pr_claim
    stale_batch = {
        "ordered_shard_ids": ["ministral3_3b_bf16-balanced-r01"],
        "execution_order": [{"shard_id": "ministral3_3b_bf16-balanced-r01"}],
        "already_completed_skips": [],
        "review_required_exclusions": [],
    }
    with pytest.raises(ResearchEvidenceError, match="refusing to reschedule settled"):
        validate_batch_against_queue(stale_batch, queue)


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


def test_stale_embedded_notebook_digest_is_bound_to_exact_frozen_source(
    tmp_path: Path,
) -> None:
    notebook = tmp_path / "frozen.ipynb"
    historical = m4_batch.HISTORICAL_STALE_NOTEBOOK_IDENTITY
    embedded = historical["embedded_stale_digest"]
    notebook_blob = (
        f"{historical['notebook_pin_commit']}:"
        "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    )
    notebook.write_bytes(
        subprocess.check_output(
            ["git", "show", notebook_blob],
            cwd=ROOT,
        )
    )
    freeze = json.loads((ROOT / "research/M4_BATCH_SOURCE_FREEZE_V7.json").read_text())
    assert freeze["batch_notebook_source_digest"] == historical[
        "batch_notebook_source_digest"
    ]
    source_identity = {"batch_notebook_source_digest": embedded}
    assert m4_batch._validate_notebook_source_identity(
        source_identity, freeze, notebook
    ) == "STALE_EMBEDDED_DIGEST_BOUND_BY_EXACT_FROZEN_SOURCE"

    source_identity["batch_notebook_source_digest"] = "d" * 64
    with pytest.raises(ResearchEvidenceError, match="source identity differs"):
        m4_batch._validate_notebook_source_identity(
            source_identity, freeze, notebook
        )
    source_identity["batch_notebook_source_digest"] = embedded
    freeze["schema_version"] = "kaggle-vllm-m4-batch-source-freeze-v9"
    with pytest.raises(ResearchEvidenceError, match="source identity differs"):
        m4_batch._validate_notebook_source_identity(
            source_identity, freeze, notebook
        )


def test_new_notebook_static_check_rejects_stale_self_digest(tmp_path: Path) -> None:
    current = ROOT / "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
    assert validate_current_notebook_self_digest(current) == m4_batch.notebook_source_digest(
        current
    )
    notebook = json.loads(current.read_text())
    bootstrap = next(cell for cell in notebook["cells"] if cell.get("id") == "bootstrap")
    bootstrap["source"] = [
        "BATCH_NOTEBOOK_SOURCE_DIGEST = '" + "0" * 64 + "'\n"
        if line.startswith("BATCH_NOTEBOOK_SOURCE_DIGEST = ")
        else line
        for line in bootstrap["source"]
    ]
    stale = tmp_path / "stale.ipynb"
    stale.write_text(json.dumps(notebook), encoding="utf-8")
    with pytest.raises(ResearchEvidenceError, match="embed its recomputed"):
        validate_current_notebook_self_digest(stale)


def test_v10_freeze_matches_generator_and_clean_notebook() -> None:
    tracked = json.loads(
        (ROOT / "research/M4_BATCH_SOURCE_FREEZE_V10.json").read_text()
    )
    assert build_freeze_v10(ROOT) == tracked
    assert verify_batch_source_freeze(
        ROOT, ROOT / "research/M4_BATCH_SOURCE_FREEZE_V10.json"
    ) == tracked
    assert tracked["protocol_amendment_version"] == "M4-BATCH-3"
    assert tracked["batch_notebook_source_digest"] == (
        "3987fe33f0b809c3f1da42846cdfd3140c0b4529f1bec8fb936febb84f5565c3"
    )


def test_v11_freeze_matches_generator_and_clean_notebook() -> None:
    tracked = json.loads(
        (ROOT / "research/M4_BATCH_SOURCE_FREEZE_V11.json").read_text()
    )
    assert build_freeze_v11(ROOT) == tracked
    assert build_freeze_v11(ROOT) == tracked
    assert verify_batch_source_freeze(
        ROOT, ROOT / "research/M4_BATCH_SOURCE_FREEZE_V11.json"
    ) == tracked
    assert tracked["protocol_amendment_version"] == "M4-BATCH-3"


def test_v12_freeze_matches_generator_and_clean_notebook() -> None:
    tracked = json.loads(
        (ROOT / "research/M4_BATCH_SOURCE_FREEZE_V12.json").read_text()
    )
    assert build_freeze_v12(ROOT) == tracked
    assert build_freeze_v12(ROOT) == tracked
    assert verify_batch_source_freeze(
        ROOT, ROOT / "research/M4_BATCH_SOURCE_FREEZE_V12.json"
    ) == tracked
    assert tracked["protocol_amendment_version"] == "M4-BATCH-3"
    assert tracked["batch_notebook_source_digest"] == (
        validate_current_notebook_self_digest(
            ROOT / "kaggle-notebooks/kaggle_vllm_m4_execute_batch.ipynb"
        )
    )
    assert tracked["implementation_source_commit"] == (
        "27fc9f3000246205e4a83fabc6244c7f4c5d98ff"
    )
    assert tracked["notebook_pin_commit"] == (
        "55d4bf61924729a297dbec787287a17d6d1b37d6"
    )


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


def test_manifest_allows_only_verified_prospectively_authorized_terminal_gate() -> None:
    batch = _batch()
    batch["protocol_amendment_version"] = "M4-BATCH-3"
    manifest = _manifest(batch, completed=3)
    resource = manifest["shards"][2]
    resource.update(
        {
            "status": "FAILED_RESOURCE_GATE",
            "runner_returncode": 3,
            "classification": "FAILED_RESOURCE_GATE",
            "reason": "VRAM_RESOURCE_GUARD",
            "post_shard_gpu_cleanup": {"status": "PASS"},
            "terminal_resource_gate": {
                "status": "VERIFIED_TERMINAL_RESOURCE_GATE",
                "classification": "FAILED_RESOURCE_GATE",
                "reason": "VRAM_RESOURCE_GUARD",
            },
        }
    )
    manifest.update(
        {
            "status": COMPLETED_WITH_TERMINAL_OUTCOMES,
            "stop_reason": None,
            "protocol_amendment_version": "M4-BATCH-3",
        }
    )
    validate_batch_manifest(
        manifest,
        batch,
        expected_source_commit="a" * 40,
        allow_terminal_resource_continuation=True,
    )
    with pytest.raises(ResearchEvidenceError, match="not authorized"):
        validate_batch_manifest(
            manifest, batch, expected_source_commit="a" * 40
        )


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


def test_historical_batch_freeze_survives_shallow_git_checkout(
    tmp_path: Path,
) -> None:
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
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=M4 test",
            "-c",
            "user.email=m4-test@example.invalid",
            "commit",
            "-m",
            "shallow fixture",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    assert verify_batch_source_freeze(repository)["protocol_amendment_version"] == (
        "M4-BATCH-1"
    )
    freeze_path = repository / "research/M4_BATCH_SOURCE_FREEZE.json"
    freeze = json.loads(freeze_path.read_text())
    freeze["batch_notebook_sha256"] = "0" * 64
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(ResearchEvidenceError, match="differs from HEAD"):
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


def _write_outer(
    root: Path,
    manifest: dict,
    inner: dict[str, bytes],
    *,
    batch_plan_sha256: str | None = None,
    batch_plan_path: str | None = None,
) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    files = {
        "BATCH_MANIFEST.json": json.dumps(manifest).encode(),
        "BATCH_SOURCE_IDENTITY.json": json.dumps(
            {
                "source_commit": "a" * 40,
                "batch_runner_sha256": "b" * 64,
                "batch_plan_sha256": batch_plan_sha256
                or sha256_file(ROOT / "research/M4_BATCH_EXECUTION_PLAN.json"),
                **({"batch_plan_path": batch_plan_path} if batch_plan_path else {}),
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
    monkeypatch.setattr(
        m4_batch,
        "_load_frozen_object",
        lambda repo, **kwargs: json.loads(
            (repo / kwargs["relative"]).read_text(encoding="utf-8")
        ),
    )
    calls = 0

    def fake_audit(**kwargs):
        nonlocal calls
        assert kwargs["maximum_runtime_delay_seconds"] == 11.5 * 3600
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
    monkeypatch.setattr(
        m4_batch,
        "_load_frozen_object",
        lambda repo, **kwargs: json.loads(
            (repo / kwargs["relative"]).read_text(encoding="utf-8")
        ),
    )

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


def _rewrite_manifest(directory: Path) -> None:
    members = sorted(
        path for path in directory.iterdir() if path.name != "SHA256SUMS.txt"
    )
    (directory / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in members),
        encoding="utf-8",
    )


def _terminal_resource_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    shard = {
        "shard_id": "qwen25_3b-prefill_heavy-r02",
        "model_key": "qwen25_3b",
        "workload": "prefill_heavy",
        "repetition": 2,
    }
    source = "a" * 40
    expected_model = {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "b" * 40,
    }
    evidence = tmp_path / "resource-gate"
    evidence.mkdir()
    runtime = {
        "completed_at": "2026-09-14T10:00:00+00:00",
        "environment": {"python": "3.12.13", "torch": "2.10.0+cu128"},
    }
    start = {
        "source": {"commit": source, "dirty": False, "dirty_paths": []},
        "started_at_utc": "2026-09-14T10:01:00+00:00",
        "runtime": [
            {},
            {"python": "3.12.13", "torch": "2.10.0+cu128", "gpu_count": 2},
        ],
        "mode": "principal",
        "sdk_version": "0.2.0",
        "model": expected_model,
        "model_key": shard["model_key"],
        "workload": shard["workload"],
        "repetition": shard["repetition"],
        "resource_limits": {"per_gpu_memory_mib": 14848.0},
    }
    rows = []
    prompt = {
        "model_id": expected_model["hf_id"],
        "model_revision": expected_model["revision"],
        "target_input_tokens": 2048,
        "prompts": [{"token_count": 2048} for _ in range(64)],
    }
    prompt_bytes = json.dumps(prompt).encode()
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    for concurrency in (1, 4, 8, 16, 32, 64):
        for tp in (1, 2):
            failed = (tp, concurrency) == (2, 64)
            rows.append(
                {
                    "model_id": expected_model["hf_id"],
                    "model_revision": expected_model["revision"],
                    "workload": shard["workload"],
                    "repetition": shard["repetition"],
                    "input_tokens": 2048,
                    "output_tokens_requested": 128,
                    "prompt_manifest_sha256": prompt_sha256,
                    "tensor_parallel_size": tp,
                    "concurrency": concurrency,
                    "oom": False,
                    "request_failures": 192 if failed else 0,
                    "maximum_vram_mib": 14895.0 if failed else 14000.0,
                    "request_throughput_per_second": None if failed else 1.0,
                    "input_tokens_per_second": None if failed else 1.0,
                    "output_tokens_per_second": None if failed else 1.0,
                    "total_tokens_per_second": None if failed else 1.0,
                    "ttft_ms": None if failed else 1.0,
                    "tpot_ms": None if failed else 1.0,
                    "itl_ms": None if failed else 1.0,
                    "e2e_latency_ms": None if failed else 1.0,
                }
            )
    failed_cell = f"{shard['shard_id']}-tp2-c64"
    message = "GPU 1 memory 14895.0 MiB exceeded 14848.0 MiB"
    objects = {
        "execution-start.json": start,
        "execution-summary.json": {
            "status": "RESOURCE_GUARD_VIOLATION",
            "expected_cells": 12,
            "completed_or_preserved_cells": 12,
            "failed_cells": 1,
            "resource_guard_violation": message,
            "semantic_gate_failure": None,
        },
        "m4-raw.json": {
            "status": "RESOURCE_GUARD_VIOLATION",
            "mode": "principal",
            "source": start["source"],
            "model_key": shard["model_key"],
            "workload": shard["workload"],
            "repetition": shard["repetition"],
            "server_lifecycle": "fresh_server_per_cell",
            "model_id": expected_model["hf_id"],
            "model_revision": expected_model["revision"],
            "prompt_manifest_sha256": prompt_sha256,
            "resource_guard_violation": message,
            "semantic_gate_failure": None,
            "rows": rows,
        },
        "terminal-resource-gate.json": {
            "schema_version": m4_batch.TERMINAL_RESOURCE_GATE_SCHEMA,
            "classification": "FAILED_RESOURCE_GATE",
            "reason": "VRAM_RESOURCE_GUARD",
            "runner_returncode": 3,
            "shard_id": shard["shard_id"],
            "failed_cell": failed_cell,
            "source_commit": source,
            "per_gpu_limit_mib": 14848.0,
            "offending_physical_gpus": [
                {"index": 0, "maximum_observed_mib": 14895.0},
                {"index": 1, "maximum_observed_mib": 14895.0},
            ],
            "resource_guard_violation": message,
            "monitor_action": "TERMINATE_CELL_PROCESS_GROUP",
            "scientific_interpretation": (
                "Frozen per-GPU VRAM boundary; failed-cell throughput is "
                "missing, not zero."
            ),
        },
        f"{failed_cell}.json": {
            "status": "executed",
            "oom_observed": False,
            "failure_observations": ["connection_error", "server_exit"],
            "identity": {"source_git_commit": source},
            "engine": {
                "model": expected_model["hf_id"],
                "model_revision": expected_model["revision"],
                "dtype": "float16",
                "tensor_parallel_size": 2,
                "gpu_memory_utilization": 0.9,
            },
            "server": {
                "unexpected_exit_returncode": 0,
                "server_log": f"{failed_cell}.server.log",
            },
            "measurements": {
                "requests": {"count": 192},
                "successful_requests": 0,
                "failed_requests": 192,
                "failure_counts": {"connection_error": 192},
                "input_tokens": 0,
                "output_tokens": 0,
                "request_throughput_per_second": 0.0,
                "input_throughput_tokens_per_second": 0.0,
                "output_throughput_tokens_per_second": 0.0,
            },
        },
    }
    for name, value in objects.items():
        (evidence / name).write_text(json.dumps(value), encoding="utf-8")
    (evidence / "prompt-manifest.json").write_bytes(prompt_bytes)
    (evidence / f"{failed_cell}.server.log").write_text(
        "Resource monitor requested cancellation\n", encoding="utf-8"
    )
    samples = [
        {"gpu_index": 0, "memory_used_mib": 14895.0},
        {"gpu_index": 1, "memory_used_mib": 14895.0},
    ]
    (evidence / f"{failed_cell}.resources.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8"
    )
    _rewrite_manifest(evidence)
    return evidence, {**shard, "expected_model": expected_model}, runtime


def test_terminal_resource_contract_accepts_realistic_qwen_monitor_exit_shape(
    tmp_path: Path,
) -> None:
    evidence, shard, runtime = _terminal_resource_fixture(tmp_path)
    audit = verify_terminal_resource_gate(
        evidence,
        expected_shard=shard,
        expected_model=shard["expected_model"],
        expected_source_commit="a" * 40,
        runtime=runtime,
        maximum_runtime_delay_seconds=3600,
    )
    assert audit == {
        "status": "VERIFIED_TERMINAL_RESOURCE_GATE",
        "classification": "FAILED_RESOURCE_GATE",
        "reason": "VRAM_RESOURCE_GUARD",
        "failed_cell": "qwen25_3b-prefill_heavy-r02-tp2-c64",
        "per_gpu_limit_mib": 14848,
        "offending_physical_gpus": [
            {"index": 0, "maximum_observed_mib": 14895.0},
            {"index": 1, "maximum_observed_mib": 14895.0},
        ],
        "gpu0_maximum_observed_mib": 14895.0,
        "gpu1_maximum_observed_mib": 14895.0,
        "failed_cell_throughput": None,
        "oom_observed": False,
    }
    raw = json.loads((evidence / "m4-raw.json").read_text())
    failed_row = next(
        row
        for row in raw["rows"]
        if row["tensor_parallel_size"] == 2 and row["concurrency"] == 64
    )
    assert failed_row["request_throughput_per_second"] is None
    assert failed_row["input_tokens_per_second"] is None
    assert failed_row["output_tokens_per_second"] is None
    assert sum(row["request_failures"] == 0 for row in raw["rows"]) == 11
    cell = json.loads(
        (evidence / "qwen25_3b-prefill_heavy-r02-tp2-c64.json").read_text()
    )
    assert cell["failure_observations"] == ["connection_error", "server_exit"]
    assert cell["server"]["unexpected_exit_returncode"] == 0


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("hash", "SHA256 mismatch"),
        ("classification", "contract mismatch"),
        ("missing_monitor_action", "contract mismatch"),
        ("missing_contract", "cannot read JSON object"),
        ("unexpected_server_exit", "unapproved operational cell failure"),
        ("oom", "reports CUDA OOM"),
        ("nccl", "CUDA OOM, NCCL, or model-load"),
        ("model_load", "CUDA OOM, NCCL, or model-load"),
    ],
)
def test_terminal_resource_contract_rejects_integrity_and_operational_failures(
    tmp_path: Path, mutation: str, error: str
) -> None:
    evidence, shard, runtime = _terminal_resource_fixture(tmp_path)
    gate_path = evidence / "terminal-resource-gate.json"
    raw_path = evidence / "m4-raw.json"
    cell_path = evidence / f"{shard['shard_id']}-tp2-c64.json"
    cell = json.loads(cell_path.read_text())
    if mutation == "hash":
        gate_path.write_text(gate_path.read_text() + " ", encoding="utf-8")
    elif mutation == "classification":
        gate = json.loads(gate_path.read_text())
        gate.pop("classification")
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        _rewrite_manifest(evidence)
    elif mutation == "missing_monitor_action":
        gate = json.loads(gate_path.read_text())
        gate.pop("monitor_action")
        gate_path.write_text(json.dumps(gate), encoding="utf-8")
        _rewrite_manifest(evidence)
    elif mutation == "missing_contract":
        gate_path.unlink()
        _rewrite_manifest(evidence)
    elif mutation == "unexpected_server_exit":
        cell["server"]["unexpected_exit_returncode"] = 1
        cell_path.write_text(json.dumps(cell), encoding="utf-8")
        _rewrite_manifest(evidence)
    elif mutation == "oom":
        raw = json.loads(raw_path.read_text())
        raw["rows"][-1]["oom"] = True
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        _rewrite_manifest(evidence)
    elif mutation in {"nccl", "model_load"}:
        marker = "NCCL error" if mutation == "nccl" else "failed to load model"
        (evidence / cell["server"]["server_log"]).write_text(marker, encoding="utf-8")
        _rewrite_manifest(evidence)
    with pytest.raises(ResearchEvidenceError, match=error):
        verify_terminal_resource_gate(
            evidence,
            expected_shard=shard,
            expected_model=shard["expected_model"],
            expected_source_commit="a" * 40,
            runtime=runtime,
            maximum_runtime_delay_seconds=3600,
        )


def _simulate_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncodes: list[int],
    verified_gate: bool = True,
    cleanup_fail_index: int | None = None,
    disk_fail_call: int | None = None,
    wall_fail_call: int | None = None,
) -> tuple[int, list[str], dict, Path]:
    repository = tmp_path / "repo"
    (repository / "scripts").mkdir(parents=True)
    (repository / "research").mkdir()
    (repository / "scripts/kaggle_m4_multimodel_crossover.py").write_text("# fixture\n")
    items = [
        {
            "shard_id": f"example-{workload}-r00",
            "model_key": "example",
            "workload": workload,
            "repetition": 0,
            "within_session_order": index,
        }
        for index, workload in enumerate(("short", "balanced", "prefill_heavy"), 1)
    ]
    batch = {
        "batch_id": "r00-example",
        "repetition": 0,
        "protocol_amendment_version": "M4-BATCH-3",
        "ordered_shard_ids": [item["shard_id"] for item in items],
        "already_completed_skips": [],
        "execution_order": items,
    }
    plan = {
        "schema_version": m4_batch.BATCH_PLAN_SCHEMA_V2,
        "terminal_resource_gate_policy": {
            "enabled": True,
            "protocol_amendment_version": "M4-BATCH-3",
            "contract_schema": m4_batch.TERMINAL_RESOURCE_GATE_SCHEMA,
            "approved_reasons": ["VRAM_RESOURCE_GUARD"],
        },
        "resource_policy": {
            "maximum_batch_wall_clock_seconds": 37800,
            "minimum_remaining_seconds_to_start_shard": 5400,
            "disk_safety_reserve_bytes": 100,
            "projected_evidence_bytes_per_shard": 100,
            "idle_gpu_memory_limit_mib_per_gpu": 10,
        },
        "batches": [batch],
    }
    plan_path = repository / "research/plan.json"
    plan_path.write_text(json.dumps(plan))
    (repository / "research/queue.json").write_text("{}")
    (repository / "research/model_matrix.json").write_text(
        json.dumps(
            {
                "models": {
                    "example": {
                        "hf_id": "example/model",
                        "selected_weight_bytes": 100,
                    }
                }
            }
        )
    )
    runtime_path = repository / "runtime.json"
    runtime_path.write_text(json.dumps({"environment": {}}))
    output_root = tmp_path / "output"
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(batch_runner, "_git_identity", lambda _repo: ("a" * 40, []))
    monkeypatch.setattr(batch_runner, "validate_batch_against_queue", lambda *_: {})
    monkeypatch.setattr(
        batch_runner,
        "_gpu_snapshot",
        lambda: {"devices": [], "compute_processes": [], "captured_at_utc": "fixture"},
    )
    cleanup_calls = 0

    def cleanup(**_kwargs):
        nonlocal cleanup_calls
        cleanup_calls += 1
        return {
            "status": "FAIL" if cleanup_calls == cleanup_fail_index else "PASS",
            "snapshot": {"devices": [], "compute_processes": []},
            "new_compute_pids": [99] if cleanup_calls == cleanup_fail_index else [],
        }

    monkeypatch.setattr(batch_runner, "_wait_for_gpu_cleanliness", cleanup)
    disk_calls = 0

    def disk(*_args, **_kwargs):
        nonlocal disk_calls
        disk_calls += 1
        if disk_calls == disk_fail_call:
            raise ResearchEvidenceError("insufficient disk before next shard")
        return {"used_bytes": disk_calls, "free_bytes": 1000, "required_free_bytes": 1}

    monkeypatch.setattr(batch_runner, "check_disk_capacity", disk)
    wall_calls = 0

    def wall(*_args, **_kwargs):
        nonlocal wall_calls
        wall_calls += 1
        if wall_calls == wall_fail_call:
            raise ResearchEvidenceError("batch wall-clock safety limit reached before next shard")
        return {"remaining_seconds": 10000}

    monkeypatch.setattr(batch_runner, "check_wall_clock", wall)
    calls: list[str] = []

    def run(command, **_kwargs):
        workload = command[command.index("--workload") + 1]
        shard_id = f"example-{workload}-r00"
        calls.append(shard_id)
        directory = output_root / "evidence" / f"{shard_id}-principal"
        directory.mkdir()
        (directory / "cell.resources.jsonl").write_text(
            json.dumps({"gpu_index": 0, "memory_used_mib": 1.0})
            + "\n"
            + json.dumps({"gpu_index": 1, "memory_used_mib": 1.0})
            + "\n"
        )
        _rewrite_manifest(directory)
        return SimpleNamespace(
            returncode=returncodes[len(calls) - 1], stdout="fixture", stderr=""
        )

    monkeypatch.setattr(batch_runner.subprocess, "run", run)

    def verify_gate(*_args, **_kwargs):
        if not verified_gate:
            raise ResearchEvidenceError("unverified resource-gate fixture")
        return {
            "status": "VERIFIED_TERMINAL_RESOURCE_GATE",
            "classification": "FAILED_RESOURCE_GATE",
            "reason": "VRAM_RESOURCE_GUARD",
            "failed_cell": "fixture",
        }

    monkeypatch.setattr(batch_runner, "verify_terminal_resource_gate", verify_gate)
    code = batch_runner.main(
        [
            "--repository",
            str(repository),
            "--output-root",
            str(output_root),
            "--runtime",
            str(runtime_path),
            "--batch-id",
            "r00-example",
            "--batch-plan",
            "research/plan.json",
            "--principal-queue",
            "research/queue.json",
            "--source-identity",
            "a" * 40,
            "--batch-notebook-source-digest",
            "b" * 64,
        ]
    )
    outer = tmp_path / "m4-batch-r00-example.zip"
    with zipfile.ZipFile(outer) as archive:
        manifest = json.loads(archive.read("BATCH_MANIFEST.json"))
        assert archive.testzip() is None
        names = set(archive.namelist())
        checksums = archive.read("BATCH_SHA256SUMS.txt").decode().splitlines()
        assert {line.split("  ", 1)[1] for line in checksums} == names - {
            "BATCH_SHA256SUMS.txt"
        }
    return code, calls, manifest, outer


@pytest.mark.parametrize(
    ("returncodes", "expected_status"),
    [
        ([0, 0, 0], "COMPLETED"),
        ([0, 3, 0], COMPLETED_WITH_TERMINAL_OUTCOMES),
        ([3, 0, 0], COMPLETED_WITH_TERMINAL_OUTCOMES),
    ],
)
def test_batch_state_machine_runs_canonical_and_verified_terminal_sequences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncodes: list[int],
    expected_status: str,
) -> None:
    code, calls, manifest, _outer = _simulate_batch(
        tmp_path, monkeypatch, returncodes=returncodes
    )
    assert code == 0
    assert calls == [
        "example-short-r00",
        "example-balanced-r00",
        "example-prefill_heavy-r00",
    ]
    assert len(calls) == len(set(calls))
    assert manifest["status"] == expected_status
    assert manifest["actual_execution_order"] == calls
    assert manifest["counts"]["failed_resource_gate"] == returncodes.count(3)


@pytest.mark.parametrize(
    ("updates", "expected_reason"),
    [
        ({"cleanup_fail_index": 2}, "GPU_CLEANLINESS_GUARD"),
        ({"verified_gate": False}, "unverified resource-gate fixture"),
        ({"disk_fail_call": 3}, "insufficient disk"),
        ({"wall_fail_call": 3}, "wall-clock"),
    ],
)
def test_batch_state_machine_fail_stops_after_unverified_or_unsafe_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict,
    expected_reason: str,
) -> None:
    code, calls, manifest, _outer = _simulate_batch(
        tmp_path,
        monkeypatch,
        returncodes=[0, 3, 0],
        **updates,
    )
    assert code == 2
    assert calls == ["example-short-r00", "example-balanced-r00"]
    assert manifest["status"] == "STOPPED_ON_FAILURE"
    assert expected_reason in manifest["stop_reason"]
    assert manifest["shards"][-1]["status"] == "NOT_EXECUTED"


@pytest.mark.parametrize("returncode", [2, 3, 4])
def test_generic_nonzero_never_continues_without_verified_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
) -> None:
    code, calls, manifest, _outer = _simulate_batch(
        tmp_path,
        monkeypatch,
        returncodes=[returncode, 0, 0],
        verified_gate=False,
    )
    assert code == 2
    assert calls == ["example-short-r00"]
    assert manifest["status"] == "STOPPED_ON_FAILURE"


def test_terminal_outcome_outer_bundle_contains_both_evidence_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _code, _calls, manifest, outer = _simulate_batch(
        tmp_path, monkeypatch, returncodes=[0, 3, 0]
    )
    with zipfile.ZipFile(outer) as archive:
        inner = {name for name in archive.namelist() if name.endswith("-principal.zip")}
    assert inner == {
        "example-short-r00-principal.zip",
        "example-balanced-r00-principal.zip",
        "example-prefill_heavy-r00-principal.zip",
    }
    assert [item["status"] for item in manifest["shards"]] == [
        "COMPLETED",
        "FAILED_RESOURCE_GATE",
        "COMPLETED",
    ]


@pytest.mark.parametrize("historical_outer_failure", [False, True])
def test_batch_ingestion_understands_terminal_outcome_and_historical_outer_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    historical_outer_failure: bool,
) -> None:
    repository = tmp_path / "repo"
    (repository / "research").mkdir(parents=True)
    (repository / "kaggle-notebooks").mkdir()
    plan = json.loads((ROOT / "research/M4_BATCH_EXECUTION_PLAN.json").read_text())
    batch = select_batch(plan, "fill-r00")
    batch["protocol_amendment_version"] = "M4-BATCH-3"
    plan["terminal_resource_gate_policy"] = {
        "enabled": True,
        "protocol_amendment_version": "M4-BATCH-3",
        "contract_schema": m4_batch.TERMINAL_RESOURCE_GATE_SCHEMA,
        "approved_reasons": ["VRAM_RESOURCE_GUARD"],
    }
    plan_path = repository / "research/prospective-plan.json"
    plan_path.write_text(json.dumps(plan))
    shutil.copy2(
        ROOT / "research/model_matrix.json",
        repository / "research/model_matrix.json",
    )
    notebook = tmp_path / "executed.ipynb"
    notebook.write_text("{}")
    runtime = tmp_path / "runtime.json"
    runtime.write_text("{}")
    resource_index = 2
    manifest = _manifest(
        batch,
        completed=(resource_index if historical_outer_failure else len(batch["ordered_shard_ids"])),
    )
    manifest.update(
        {
            "protocol_amendment_version": "M4-BATCH-3",
            "status": (
                "STOPPED_ON_FAILURE"
                if historical_outer_failure
                else COMPLETED_WITH_TERMINAL_OUTCOMES
            ),
            "stop_reason": (
                "terminal resource gate includes an unapproved operational cell failure"
                if historical_outer_failure
                else None
            ),
        }
    )
    resource_id = batch["ordered_shard_ids"][resource_index - 1]
    resource = next(
        item for item in manifest["shards"] if item["shard_id"] == resource_id
    )
    resource.update(
        {
            "status": "FAILED" if historical_outer_failure else "FAILED_RESOURCE_GATE",
            "runner_returncode": 3,
            "classification": None if historical_outer_failure else "FAILED_RESOURCE_GATE",
            "reason": (
                "terminal resource gate includes an unapproved operational cell failure"
                if historical_outer_failure
                else "VRAM_RESOURCE_GUARD"
            ),
            "post_shard_gpu_cleanup": {"status": "PASS"},
        }
    )
    if not historical_outer_failure:
        resource["terminal_resource_gate"] = {
            "status": "VERIFIED_TERMINAL_RESOURCE_GATE",
            "classification": "FAILED_RESOURCE_GATE",
            "reason": "VRAM_RESOURCE_GUARD",
        }
    inner = {
        f"{shard_id}-principal.zip": shard_id.encode()
        for shard_id in batch["ordered_shard_ids"]
    }
    for item in manifest["shards"]:
        if item["status"] not in {"COMPLETED", "FAILED", "FAILED_RESOURCE_GATE"}:
            continue
        item["evidence_zip_sha256"] = hashlib.sha256(
            inner[item["evidence_zip"]]
        ).hexdigest()
    plan_sha = sha256_file(plan_path)
    outer = _write_outer(
        tmp_path,
        manifest,
        inner,
        batch_plan_sha256=plan_sha,
        batch_plan_path="research/prospective-plan.json",
    )
    monkeypatch.setattr(
        m4_batch,
        "select_batch_source_freeze",
        lambda _repository, _source: (
            repository / "research/M4_BATCH_SOURCE_FREEZE_V9.json",
            {
                "implementation_source_commit": "a" * 40,
                "notebook_pin_commit": "b" * 40,
                "batch_runner_sha256": "b" * 64,
                "batch_plan_sha256": plan_sha,
                    "batch_plan_path": "research/prospective-plan.json",
                    "batch_notebook_source_digest": "c" * 64,
                    "model_matrix_sha256": sha256_file(
                        repository / "research/model_matrix.json"
                    ),
            },
        ),
    )
    monkeypatch.setattr(m4_batch, "_git_blob", lambda *_args: b"{}")
    monkeypatch.setattr(m4_batch, "notebook_sources", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(m4_batch, "verify_runtime", lambda _runtime: None)
    monkeypatch.setattr(
        m4_batch,
        "_load_frozen_object",
        lambda repo, **kwargs: json.loads(
            (repo / kwargs["relative"]).read_text(encoding="utf-8")
        ),
    )

    def fake_audit(**kwargs):
        shard_id = kwargs["evidence_zip"].name.removesuffix("-principal.zip")
        extracted = tmp_path / f"extracted-{shard_id}"
        extracted.mkdir()
        (extracted / "evidence.txt").write_text("validated fixture")
        outcome = next(
            item for item in manifest["shards"] if item["shard_id"] == shard_id
        )
        (extracted / "batch-shard-provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": "kaggle-vllm-m4-batch-shard-provenance-v1",
                    "execution_mode": "batch_orchestrated",
                    "shard_id": shard_id,
                    "batch_id": manifest["batch_id"],
                    "session_id": manifest["session_id"],
                    "repetition": manifest["repetition"],
                    "within_session_order": outcome["within_session_order"],
                    "source_commit": "a" * 40,
                }
            )
        )
        return (
            {
                "shard_id": shard_id,
                "evidence_zip_sha256": sha256_file(kwargs["evidence_zip"]),
                "status": "VERIFIED_CANONICAL_CANDIDATE",
            },
            extracted,
        )

    monkeypatch.setattr(m4_batch, "audit_download", fake_audit)
    monkeypatch.setattr(
        m4_batch,
        "_audit_terminal_resource_archive",
        lambda **_kwargs: {
            "shard_id": resource_id,
            "status": "VERIFIED_TERMINAL_RESOURCE_GATE",
            "classification": "FAILED_RESOURCE_GATE",
            "reason": "VRAM_RESOURCE_GUARD",
        },
    )
    report = stage_batch_download(
        repository=repository,
        notebook=notebook,
        batch_zip=outer,
        runtime_path=runtime,
    )
    assert report["status"] == "BATCH_REVIEW_REQUIRED"
    assert report["batch_manifest_status"] == (
        "STOPPED_ON_FAILURE"
        if historical_outer_failure
        else COMPLETED_WITH_TERMINAL_OUTCOMES
    )
    assert len(report["accepted_shards"]) == (
        resource_index - 1 if historical_outer_failure else 10
    )
    assert report["invalid_shards"] == []
    assert report["failed_shards"] == []
    assert [item["shard_id"] for item in report["terminal_resource_shards"]] == [
        resource_id
    ]
    if historical_outer_failure:
        terminal = report["terminal_resource_shards"][0]
        assert terminal["historical_outer_outcome_status"] == "FAILED"
        assert terminal["historical_outer_reason"] == (
            "terminal resource gate includes an unapproved operational cell failure"
        )
