"""Fail-closed local audit and staging for downloaded M4 evidence."""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ResearchEvidenceError
from .provenance import sha256_file, verify_sha256_manifest

EXPECTED_SOURCE_COMMIT = "42bf096c032e2c6be1e2fa3d573c7c86ac589ba2"
EXPECTED_WHEEL_SHA256 = "5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c"
EXPECTED_PROFILE = "kaggle-t4x2-cu128"
EXPECTED_VERSION = "0.2.0"
CONCURRENCY = [1, 4, 8, 16, 32, 64]


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchEvidenceError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ResearchEvidenceError(f"expected JSON object: {path}")
    return value


def notebook_sources(path: Path) -> list[tuple[Any, Any, str]]:
    notebook = _object(path)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ResearchEvidenceError(f"notebook has no cells: {path}")
    return [
        (cell.get("cell_type"), cell.get("id"), "".join(cell.get("source", [])))
        for cell in cells
    ]


def inspect_zip(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            bad_crc = archive.testzip()
    except (OSError, zipfile.BadZipFile) as error:
        raise ResearchEvidenceError(f"invalid evidence ZIP: {path}") from error
    if bad_crc is not None:
        raise ResearchEvidenceError(f"ZIP CRC failure: {bad_crc}")
    names: set[str] = set()
    for info in infos:
        name = info.filename
        relative = PurePosixPath(name)
        mode = info.external_attr >> 16
        if name in names:
            raise ResearchEvidenceError(f"duplicate ZIP member: {name}")
        if relative.is_absolute() or ".." in relative.parts:
            raise ResearchEvidenceError(f"unsafe ZIP path: {name}")
        if info.flag_bits & 0x1:
            raise ResearchEvidenceError(f"encrypted ZIP member: {name}")
        if stat.S_ISLNK(mode):
            raise ResearchEvidenceError(f"symlink ZIP member: {name}")
        if len(relative.parts) != 1 or info.is_dir():
            raise ResearchEvidenceError(f"M4 evidence must be a flat file archive: {name}")
        names.add(name)
    required = {"SHA256SUMS.txt", "execution-start.json", "execution-summary.json", "m4-raw.json", "prompt-manifest.json"}
    missing = required - names
    if missing:
        raise ResearchEvidenceError(f"evidence ZIP lacks required members: {sorted(missing)}")
    return sorted(names)


def verify_runtime(runtime: dict[str, Any]) -> None:
    environment = runtime.get("environment", {})
    wheel = runtime.get("wheel", {})
    gpus = environment.get("gpus", [])
    checks = {
        "profile": runtime.get("profile") == EXPECTED_PROFILE,
        "strict": runtime.get("strict") is True,
        "python": environment.get("python") == "3.12.13",
        "torch": environment.get("torch") == "2.10.0+cu128",
        "torch_cuda": environment.get("torch_cuda") == "12.8",
        "cuda": environment.get("cuda_available") is True,
        "nccl": environment.get("nccl") == "2.27.5",
        "driver": environment.get("driver_version") == "580.159.04",
        "cuda_toolkit": "V12.8.93" in str(environment.get("nvcc_version")),
        "wheel": wheel.get("sha256") == EXPECTED_WHEEL_SHA256,
        "wheel_repository": wheel.get("hf_repo_id") == "waqasm86/kaggle-vllm-binaries",
        "wheel_revision": wheel.get("hf_revision") == "f6b4f10de54924ed6fe9e28cceab84eca7276ab6",
        "gpus": len(gpus) == 2
        and all(gpu.get("name") == "Tesla T4" and gpu.get("capability") == [7, 5] for gpu in gpus),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ResearchEvidenceError(f"runtime identity mismatch: {failed}")


def _expected_shard(start: dict[str, Any]) -> str:
    if start.get("mode") == "compatibility":
        return f"compat-{start.get('model_key')}"
    repetition = start.get("repetition")
    if start.get("mode") == "principal" and isinstance(repetition, int):
        return f"{start.get('model_key')}-{start.get('workload')}-r{repetition:02d}"
    raise ResearchEvidenceError("only compatibility and principal M4 shards are ingestible")


def audit_download(
    *, repository: Path, notebook: Path, evidence_zip: Path, runtime_path: Path
) -> tuple[dict[str, Any], Path]:
    frozen = repository / "kaggle-notebooks/kaggle_vllm_m4_execute_shard.ipynb"
    if notebook_sources(notebook) != notebook_sources(frozen):
        raise ResearchEvidenceError("executed notebook source differs from frozen source")
    members = inspect_zip(evidence_zip)
    runtime = _object(runtime_path)
    verify_runtime(runtime)
    temporary = Path(tempfile.mkdtemp(prefix="kaggle-vllm-m4-ingest-"))
    try:
        with zipfile.ZipFile(evidence_zip) as archive:
            archive.extractall(temporary)
        verified = verify_sha256_manifest(temporary, temporary / "SHA256SUMS.txt")
        start = _object(temporary / "execution-start.json")
        summary = _object(temporary / "execution-summary.json")
        raw = _object(temporary / "m4-raw.json")
        prompt = _object(temporary / "prompt-manifest.json")
        source = start.get("source", {})
        model = start.get("model", {})
        matrix = _object(repository / "research/model_matrix.json")["models"]
        model_key = start.get("model_key")
        expected_model = matrix.get(model_key)
        if expected_model is None:
            raise ResearchEvidenceError(f"unknown M4 model key: {model_key}")
        if source.get("commit") != EXPECTED_SOURCE_COMMIT or source.get("dirty") is not False:
            raise ResearchEvidenceError("evidence does not use the clean frozen source commit")
        if start.get("sdk_version") != EXPECTED_VERSION:
            raise ResearchEvidenceError("evidence SDK version is not 0.2.0")
        if (
            model.get("hf_id") != expected_model.get("hf_id")
            or model.get("revision") != expected_model.get("revision")
            or model.get("architecture") != expected_model.get("architecture")
        ):
            raise ResearchEvidenceError("model identity differs from the frozen model matrix")
        if raw.get("model_id") != model.get("hf_id") or raw.get("model_revision") != model.get("revision"):
            raise ResearchEvidenceError("raw/model identity mismatch")
        if prompt.get("model_id") != model.get("hf_id") or prompt.get("model_revision") != model.get("revision"):
            raise ResearchEvidenceError("prompt/model identity mismatch")
        shard_id = _expected_shard(start)
        plan = _object(repository / "research/M4_EXECUTION_PLAN.json")
        planned = {
            row["shard_id"]
            for key in ("compatibility_order", "principal_order")
            for row in plan[key]
        }
        if shard_id not in planned:
            raise ResearchEvidenceError(f"shard is absent from the frozen plan: {shard_id}")
        mode = start["mode"]
        expected_concurrency = [1] if mode == "compatibility" else CONCURRENCY
        expected_cells = len(expected_concurrency) * 2
        if summary.get("expected_cells") != expected_cells or len(raw.get("rows", [])) != expected_cells:
            raise ResearchEvidenceError("shard cell count does not match the frozen protocol")
        identities = {(row.get("tensor_parallel_size"), row.get("concurrency")) for row in raw["rows"]}
        if identities != {(tp, concurrency) for tp in (1, 2) for concurrency in expected_concurrency}:
            raise ResearchEvidenceError("TP/concurrency grid does not match the frozen protocol")
        token_targets = {"short": (128, 64), "balanced": (512, 256), "prefill_heavy": (2048, 128)}
        expected_tokens = token_targets.get(start.get("workload"))
        if expected_tokens is None or any(
            (row.get("input_tokens"), row.get("output_tokens_requested")) != expected_tokens
            for row in raw["rows"]
        ):
            raise ResearchEvidenceError("row token targets do not match the frozen workload")
        prompts = prompt.get("prompts")
        if (
            prompt.get("target_input_tokens") != expected_tokens[0]
            or not isinstance(prompts, list)
            or len(prompts) != 64
            or any(item.get("token_count") != expected_tokens[0] for item in prompts)
        ):
            raise ResearchEvidenceError("prompt manifest is not the 64-prompt exact-token corpus")
        if any(
            (row.get("maximum_vram_mib") is not None and row["maximum_vram_mib"] > 14848)
            or (
                row.get("maximum_system_ram_bytes") is not None
                and row["maximum_system_ram_bytes"] > 30064771072
            )
            for row in raw["rows"]
        ):
            raise ResearchEvidenceError("M4 resource limit was exceeded")
        for tp, concurrency in identities:
            stem = f"{model_key}-{start['workload']}-r{start['repetition']:02d}-tp{tp}-c{concurrency:02d}"
            result = _object(temporary / f"{stem}.json")
            engine = result.get("engine", {})
            if (
                engine.get("model") != expected_model["hf_id"]
                or engine.get("model_revision") != expected_model["revision"]
                or engine.get("dtype") != "float16"
                or engine.get("tensor_parallel_size") != tp
                or result.get("concurrency") != concurrency
            ):
                raise ResearchEvidenceError(f"cell identity/configuration mismatch: {stem}")
        failed = summary.get("failed_cells", 0)
        if mode == "compatibility" and failed == 0:
            classification = "COMPATIBILITY_PASS"
        elif start.get("model_key") == "gemma3_4b" and failed == 2:
            logs = "\n".join(
                (temporary / f"gemma3_4b-short-r00-tp{tp}-c01.server.log").read_text(errors="replace")
                for tp in (1, 2)
            )
            phrases = ("Gemma3ForConditionalGeneration", "does not support float16", "Numerical instability")
            if not all(logs.count(phrase) >= 2 for phrase in phrases):
                raise ResearchEvidenceError("Gemma failure does not reproduce the frozen dtype guard in both logs")
            if any(row.get("request_throughput_per_second") is not None for row in raw["rows"]):
                raise ResearchEvidenceError("failed Gemma cell contains a throughput value")
            classification = "UNSUPPORTED_DTYPE_INTERSECTION_ON_SM75_FROZEN_STACK"
        elif mode == "principal":
            classification = "PRINCIPAL_SHARD_PRESERVED"
        else:
            classification = "COMPATIBILITY_FAILURE_REQUIRES_MANUAL_CLASSIFICATION"
        audit = {
            "schema_version": "kaggle-vllm-m4-ingest-audit-v1",
            "status": "VERIFIED_CANONICAL_CANDIDATE",
            "classification": classification,
            "shard_id": shard_id,
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "source_equivalent_notebook": True,
            "evidence_zip_sha256": sha256_file(evidence_zip),
            "executed_notebook_sha256": sha256_file(notebook),
            "runtime_json_sha256": sha256_file(runtime_path),
            "verified_payload_count": len(verified),
            "zip_member_count": len(members),
            "destination_status": "LOCAL_STAGING_ONLY_REQUIRES_REVIEWED_PROMOTION",
        }
        return audit, temporary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def stage_download(
    *, repository: Path, notebook: Path, evidence_zip: Path, runtime_path: Path
) -> dict[str, Any]:
    audit, extracted = audit_download(
        repository=repository,
        notebook=notebook,
        evidence_zip=evidence_zip,
        runtime_path=runtime_path,
    )
    destination = repository / ".local-evidence" / "m4-ingest" / audit["evidence_zip_sha256"]
    if destination.exists():
        shutil.rmtree(extracted)
        raise ResearchEvidenceError(f"refusing duplicate/conflicting ingest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted), destination)
    originals = destination / "original"
    originals.mkdir()
    for source in (notebook, evidence_zip, runtime_path):
        shutil.copy2(source, originals / source.name)
    (destination / "INGEST_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    audit["staged_at"] = str(destination)
    return audit
