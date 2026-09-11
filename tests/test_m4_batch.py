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


def test_partial_manifest_and_duplicate_shards() -> None:
    batch = _batch()
    manifest = _manifest(batch, completed=2)
    validate_batch_manifest(manifest, batch, expected_source_commit="a" * 40)
    manifest["shards"][-1]["shard_id"] = manifest["shards"][-2]["shard_id"]
    with pytest.raises(ResearchEvidenceError, match="duplicate shard IDs"):
        validate_batch_manifest(manifest, batch, expected_source_commit="a" * 40)


def _write_outer(root: Path, manifest: dict, inner: dict[str, bytes]) -> Path:
    bundle = root / "bundle"
    bundle.mkdir()
    files = {
        "BATCH_MANIFEST.json": json.dumps(manifest).encode(),
        "BATCH_SOURCE_IDENTITY.json": json.dumps(
            {"source_commit": "a" * 40, "batch_runner_sha256": "b" * 64}
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
        "verify_batch_source_freeze",
        lambda _repository: {
            "implementation_source_commit": "a" * 40,
            "batch_runner_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(m4_batch, "notebook_sources", lambda _path: [])
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
        return (
            {
                "shard_id": manifest["actual_execution_order"][0],
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
