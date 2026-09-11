"""Fail-closed local audit and staging for downloaded M4 evidence."""

from __future__ import annotations

import json
import math
import shutil
import stat
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import ResearchEvidenceError
from .provenance import sha256_file, verify_sha256_manifest
from .resources import GPU_MEMORY_LIMIT_MIB, SYSTEM_RAM_LIMIT_BYTES

EXPECTED_SOURCE_COMMIT = "42bf096c032e2c6be1e2fa3d573c7c86ac589ba2"
EXPECTED_WHEEL_SHA256 = "5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c"
EXPECTED_PROFILE = "kaggle-t4x2-cu128"
EXPECTED_VERSION = "0.2.0"
CONCURRENCY = [1, 4, 8, 16, 32, 64]
MAX_ZIP_MEMBERS = 1_000
MAX_ZIP_MEMBER_BYTES = 512 * 1024**2
MAX_ZIP_UNCOMPRESSED_BYTES = 2 * 1024**3
SUCCESS_METRICS = (
    "request_throughput_per_second",
    "input_tokens_per_second",
    "output_tokens_per_second",
    "total_tokens_per_second",
    "ttft_ms",
    "tpot_ms",
    "itl_ms",
    "e2e_latency_ms",
    "gpu_utilization_percent",
    "maximum_vram_mib",
    "maximum_system_ram_bytes",
    "mean_power_w",
    "maximum_temperature_c",
)


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
            if len(infos) > MAX_ZIP_MEMBERS:
                raise ResearchEvidenceError("M4 evidence ZIP has too many members")
            total_size = 0
            names: set[str] = set()
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
                ):
                    raise ResearchEvidenceError(f"unsafe ZIP path: {name}")
                if name in names:
                    raise ResearchEvidenceError(f"duplicate ZIP member: {name}")
                if info.flag_bits & 0x1:
                    raise ResearchEvidenceError(f"encrypted ZIP member: {name}")
                if stat.S_ISLNK(mode):
                    raise ResearchEvidenceError(f"symlink ZIP member: {name}")
                if len(relative.parts) != 1 or info.is_dir():
                    raise ResearchEvidenceError(
                        f"M4 evidence must be a flat file archive: {name}"
                    )
                if info.file_size > MAX_ZIP_MEMBER_BYTES:
                    raise ResearchEvidenceError(f"oversized ZIP member: {name}")
                total_size += info.file_size
                if total_size > MAX_ZIP_UNCOMPRESSED_BYTES:
                    raise ResearchEvidenceError(
                        "M4 evidence ZIP exceeds the uncompressed size limit"
                    )
                names.add(name)
            bad_crc = archive.testzip()
    except (OSError, zipfile.BadZipFile) as error:
        raise ResearchEvidenceError(f"invalid evidence ZIP: {path}") from error
    if bad_crc is not None:
        raise ResearchEvidenceError(f"ZIP CRC failure: {bad_crc}")
    required = {
        "SHA256SUMS.txt",
        "execution-start.json",
        "execution-summary.json",
        "m4-raw.json",
        "prompt-manifest.json",
    }
    missing = required - names
    if missing:
        raise ResearchEvidenceError(f"evidence ZIP lacks required members: {sorted(missing)}")
    return sorted(names)


def verify_runtime(runtime: dict[str, Any]) -> None:
    environment = runtime.get("environment", {})
    wheel = runtime.get("wheel", {})
    if not isinstance(environment, dict) or not isinstance(wheel, dict):
        raise ResearchEvidenceError("runtime identity sections must be JSON objects")
    gpus = environment.get("gpus", [])
    if not isinstance(gpus, list) or any(not isinstance(gpu, dict) for gpu in gpus):
        raise ResearchEvidenceError("runtime GPU identity must be a JSON list of objects")
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


def _verify_runtime_binding(runtime: dict[str, Any], start: dict[str, Any]) -> None:
    captured = start.get("runtime")
    if not isinstance(captured, list) or len(captured) < 2:
        raise ResearchEvidenceError("execution-start lacks captured runtime identity")
    manifest_environment = runtime.get("environment", {})
    execution_environment = captured[1]
    if not isinstance(execution_environment, dict) or any(
        execution_environment.get(key) != value
        for key, value in manifest_environment.items()
    ):
        raise ResearchEvidenceError("runtime.json differs from execution-start runtime")
    try:
        completed = datetime.fromisoformat(str(runtime["completed_at"]))
        started = datetime.fromisoformat(str(start["started_at_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ResearchEvidenceError("runtime/execution timestamps are invalid") from error
    delay = (started - completed).total_seconds()
    if delay < 0 or delay > 3600:
        raise ResearchEvidenceError(
            "runtime.json is not temporally bound to the shard execution"
        )


def _expected_shard(start: dict[str, Any]) -> str:
    if start.get("mode") == "compatibility":
        return f"compat-{start.get('model_key')}"
    repetition = start.get("repetition")
    if start.get("mode") == "principal" and isinstance(repetition, int):
        return f"{start.get('model_key')}-{start.get('workload')}-r{repetition:02d}"
    raise ResearchEvidenceError("only compatibility and principal M4 shards are ingestible")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ResearchEvidenceError(
                    f"expected JSON object at {path}:{line_number}"
                )
            rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchEvidenceError(f"cannot read JSONL evidence: {path}") from error
    return rows


def _positive_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchEvidenceError(f"{field} must be a positive finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ResearchEvidenceError(f"{field} must be a positive finite number")
    return normalized


def _nonnegative_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchEvidenceError(f"{field} must be a nonnegative finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ResearchEvidenceError(f"{field} must be a nonnegative finite number")
    return normalized


def _validate_successful_cell(
    *,
    root: Path,
    result: dict[str, Any],
    row: dict[str, Any],
    stem: str,
    concurrency: int,
    input_tokens: int,
    output_tokens: int,
    prompt_sha256: str,
) -> None:
    expected_requests = max(20, concurrency * 3)
    measurements = result.get("measurements", {})
    workload = result.get("workload", {})
    requests = measurements.get("requests", {})
    request_name = f"{stem}-requests.jsonl"
    checks = {
        "cell status": result.get("status") == "executed",
        "workload concurrency": workload.get("concurrency") == concurrency,
        "warmup count": workload.get("warmup_requests") == concurrency,
        "request count": workload.get("total_requests") == expected_requests,
        "output target": workload.get("max_output_tokens") == output_tokens,
        "ignore EOS": workload.get("ignore_eos") is True,
        "prefix cache": result.get("engine", {}).get("enable_prefix_caching") is False,
        "prompt hash": workload.get("prompt_manifest_sha256") == prompt_sha256,
        "successful requests": measurements.get("successful_requests") == expected_requests,
        "failed requests": measurements.get("failed_requests") == 0,
        "request ledger filename": requests.get("file") == request_name,
        "request ledger count": requests.get("count") == expected_requests,
        "request ledger format": requests.get("storage") == "jsonl",
        "row measured requests": row.get("measured_requests") == expected_requests,
        "row warmups": row.get("warmup_requests") == concurrency,
        "row failures": row.get("request_failures") == 0,
        "row OOM": row.get("oom") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ResearchEvidenceError(f"successful-cell invariant mismatch for {stem}: {failed}")
    for field in SUCCESS_METRICS:
        _positive_number(row.get(field), field=f"{stem}.{field}")
    if measurements.get("input_tokens") != expected_requests * input_tokens:
        raise ResearchEvidenceError(f"aggregate input-token mismatch: {stem}")
    if measurements.get("output_tokens") != expected_requests * output_tokens:
        raise ResearchEvidenceError(f"aggregate output-token mismatch: {stem}")
    for name, target in (
        ("input_tokens_per_request", input_tokens),
        ("output_tokens_per_request", output_tokens),
    ):
        distribution = measurements.get(name, {})
        if (
            distribution.get("count") != expected_requests
            or distribution.get("min") != target
            or distribution.get("max") != target
        ):
            raise ResearchEvidenceError(f"per-request token distribution mismatch: {stem}")
    request_rows = _jsonl(root / request_name)
    if len(request_rows) != expected_requests:
        raise ResearchEvidenceError(f"request ledger length mismatch: {stem}")
    request_ids = {request.get("request_id") for request in request_rows}
    if len(request_ids) != expected_requests or None in request_ids:
        raise ResearchEvidenceError(f"request ledger IDs are missing or duplicated: {stem}")
    if any(
        request.get("status") != "completed"
        or request.get("concurrency") != concurrency
        or request.get("input_tokens") != input_tokens
        or request.get("output_tokens") != output_tokens
        for request in request_rows
    ):
        raise ResearchEvidenceError(f"request ledger semantic mismatch: {stem}")
    resource_rows = _jsonl(root / f"{stem}.resources.jsonl")
    if not resource_rows:
        raise ResearchEvidenceError(f"resource ledger is empty: {stem}")
    resource_gpus: dict[int, list[float]] = {0: [], 1: []}
    maximum_system_ram_bytes = 0.0
    for sample_number, sample in enumerate(resource_rows, 1):
        gpu_index = sample.get("gpu_index")
        if gpu_index not in resource_gpus or sample.get("phase") != stem:
            raise ResearchEvidenceError(
                f"resource ledger GPU/phase mismatch: {stem}:{sample_number}"
            )
        memory_mib = _nonnegative_number(
            sample.get("memory_used_mib"),
            field=f"{stem}.resources[{sample_number}].memory_used_mib",
        )
        system_ram_bytes = _nonnegative_number(
            sample.get("system_used_bytes"),
            field=f"{stem}.resources[{sample_number}].system_used_bytes",
        )
        resource_gpus[gpu_index].append(memory_mib)
        maximum_system_ram_bytes = max(maximum_system_ram_bytes, system_ram_bytes)
        if memory_mib > GPU_MEMORY_LIMIT_MIB:
            raise ResearchEvidenceError(
                f"per-GPU VRAM limit exceeded: {stem} GPU{gpu_index} "
                f"{memory_mib} MiB > {GPU_MEMORY_LIMIT_MIB} MiB"
            )
        if system_ram_bytes > SYSTEM_RAM_LIMIT_BYTES:
            raise ResearchEvidenceError(
                f"system RAM limit exceeded: {stem} {system_ram_bytes} bytes"
            )
    if any(not samples for samples in resource_gpus.values()):
        raise ResearchEvidenceError(
            f"resource ledger does not cover both physical GPUs: {stem}"
        )
    visible_gpus = result.get("server", {}).get("visible_physical_gpu_indices")
    if visible_gpus not in ([0], [0, 1]):
        raise ResearchEvidenceError(f"visible physical GPU identity is invalid: {stem}")
    recomputed_vram = max(
        memory_mib for gpu_index in visible_gpus for memory_mib in resource_gpus[gpu_index]
    )
    if row.get("maximum_vram_mib") != recomputed_vram:
        raise ResearchEvidenceError(f"raw/resource VRAM summary mismatch: {stem}")
    if row.get("maximum_system_ram_bytes") != maximum_system_ram_bytes:
        raise ResearchEvidenceError(f"raw/resource RAM summary mismatch: {stem}")

    telemetry_rows = _jsonl(root / f"{stem}.telemetry.jsonl")
    if not telemetry_rows:
        raise ResearchEvidenceError(f"telemetry ledger is empty: {stem}")
    telemetry = result.get("gpu_telemetry", {})
    if not telemetry.get("telemetry_sample_count") or not telemetry.get("summaries"):
        raise ResearchEvidenceError(f"GPU telemetry summary is absent: {stem}")
    telemetry_gpus: dict[int, list[float]] = {0: [], 1: []}
    for sample_number, sample in enumerate(telemetry_rows, 1):
        gpu_index = sample.get("index")
        if gpu_index not in telemetry_gpus:
            raise ResearchEvidenceError(
                f"telemetry GPU identity mismatch: {stem}:{sample_number}"
            )
        memory_mib = _nonnegative_number(
            sample.get("memory_used_mib"),
            field=f"{stem}.telemetry[{sample_number}].memory_used_mib",
        )
        telemetry_gpus[gpu_index].append(memory_mib)
        if memory_mib > GPU_MEMORY_LIMIT_MIB:
            raise ResearchEvidenceError(
                f"per-GPU VRAM limit exceeded: {stem} GPU{gpu_index} "
                f"{memory_mib} MiB > {GPU_MEMORY_LIMIT_MIB} MiB"
            )
    summaries = telemetry["summaries"]
    if (
        {item.get("index") for item in summaries} != {0, 1}
        or telemetry.get("telemetry_sample_count") != len(telemetry_rows)
    ):
        raise ResearchEvidenceError(f"GPU telemetry coverage mismatch: {stem}")
    for summary in summaries:
        gpu_index = summary["index"]
        if summary.get("peak_memory_used_mib") != max(telemetry_gpus[gpu_index]):
            raise ResearchEvidenceError(f"GPU telemetry peak mismatch: {stem} GPU{gpu_index}")


def _check_existing_shard(repository: Path, shard_id: str, destination: Path) -> None:
    staging = repository / ".local-evidence" / "m4-ingest"
    if destination.exists():
        raise ResearchEvidenceError(f"refusing duplicate/conflicting ingest: {destination}")
    if not staging.exists():
        return
    for child in staging.iterdir():
        if not child.is_dir():
            raise ResearchEvidenceError(f"unexpected M4 staging entry: {child}")
        audit_path = child / "INGEST_AUDIT.json"
        try:
            existing = _object(audit_path)
        except ResearchEvidenceError as error:
            raise ResearchEvidenceError(
                f"cannot prove M4 staging conflict safety: {audit_path}"
            ) from error
        if existing.get("shard_id") == shard_id:
            raise ResearchEvidenceError(
                f"refusing second candidate for already staged shard {shard_id}: {child}"
            )


def audit_download(
    *,
    repository: Path,
    notebook: Path,
    evidence_zip: Path,
    runtime_path: Path,
    expected_source_commit: str = EXPECTED_SOURCE_COMMIT,
    frozen_notebook: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    frozen = frozen_notebook or (
        repository / "kaggle-notebooks/kaggle_vllm_m4_execute_shard.ipynb"
    )
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
        if set(verified) != set(members) - {"SHA256SUMS.txt"}:
            raise ResearchEvidenceError(
                "SHA256SUMS.txt must cover every non-manifest ZIP member exactly"
            )
        start = _object(temporary / "execution-start.json")
        _verify_runtime_binding(runtime, start)
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
        if source.get("commit") != expected_source_commit or source.get("dirty") is not False:
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
            row["shard_id"]: row
            for key in ("compatibility_order", "principal_order")
            for row in plan[key]
        }
        if shard_id not in planned:
            raise ResearchEvidenceError(f"shard is absent from the frozen plan: {shard_id}")
        mode = start["mode"]
        planned_row = planned[shard_id]
        if any(
            start.get(field) != planned_row.get(field)
            for field in ("mode", "model_key", "workload", "repetition")
        ):
            raise ResearchEvidenceError("shard fields differ from the frozen plan")
        if mode == "principal":
            evidence_status = _object(repository / "research/M4_EVIDENCE_STATUS.json")
            compatibility = evidence_status.get("compatibility", {}).get(model_key, {})
            if compatibility.get("status") != "COMPATIBILITY_PASS":
                raise ResearchEvidenceError(
                    "principal shard model did not pass the frozen compatibility gate"
                )
        expected_concurrency = [1] if mode == "compatibility" else CONCURRENCY
        expected_cells = len(expected_concurrency) * 2
        if summary.get("expected_cells") != expected_cells or len(raw.get("rows", [])) != expected_cells:
            raise ResearchEvidenceError("shard cell count does not match the frozen protocol")
        if (
            summary.get("completed_or_preserved_cells") != expected_cells
            or raw.get("mode") != mode
            or raw.get("model_key") != model_key
            or raw.get("workload") != start.get("workload")
            or raw.get("repetition") != start.get("repetition")
            or raw.get("source") != start.get("source")
            or raw.get("server_lifecycle") != "fresh_server_per_cell"
        ):
            raise ResearchEvidenceError("raw/summary shard identity is incomplete or inconsistent")
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
        prompt_sha256 = sha256_file(temporary / "prompt-manifest.json")
        if (
            raw.get("prompt_manifest_sha256") != prompt_sha256
            or any(row.get("prompt_manifest_sha256") != prompt_sha256 for row in raw["rows"])
        ):
            raise ResearchEvidenceError("prompt manifest hash is not bound to every raw row")
        if any(
            (
                row.get("maximum_vram_mib") is not None
                and row["maximum_vram_mib"] > GPU_MEMORY_LIMIT_MIB
            )
            or (
                row.get("maximum_system_ram_bytes") is not None
                and row["maximum_system_ram_bytes"] > SYSTEM_RAM_LIMIT_BYTES
            )
            for row in raw["rows"]
        ):
            raise ResearchEvidenceError("M4 resource limit was exceeded")
        if mode == "principal" and (
            summary.get("status") != "executed"
            or raw.get("status") != "executed"
            or summary.get("failed_cells") != 0
            or summary.get("resource_guard_violation") is not None
            or summary.get("semantic_gate_failure") is not None
            or raw.get("resource_guard_violation") is not None
            or raw.get("semantic_gate_failure") is not None
        ):
            raise ResearchEvidenceError(
                "principal shard is not a complete zero-unexpected-failure execution"
            )
        for tp, concurrency in identities:
            stem = f"{model_key}-{start['workload']}-r{start['repetition']:02d}-tp{tp}-c{concurrency:02d}"
            result = _object(temporary / f"{stem}.json")
            engine = result.get("engine", {})
            identity = result.get("identity", {})
            workload = result.get("workload", {})
            if (
                engine.get("model") != expected_model["hf_id"]
                or engine.get("model_revision") != expected_model["revision"]
                or engine.get("dtype") != "float16"
                or engine.get("tensor_parallel_size") != tp
                or engine.get("model_representation") != "transformers"
                or engine.get("model_source") != "huggingface"
                or engine.get("served_model_name") != model_key
                or engine.get("api_mode") != "completions"
                or identity.get("source_git_commit") != expected_source_commit
                or identity.get("kaggle_vllm_version") != EXPECTED_VERSION
                or workload.get("prompt_profile")
                != "M4 exact tokenizer-controlled completion prompts v1"
                or workload.get("seed") != start["repetition"]
                or workload.get("temperature") != 0.0
                or workload.get("retain_prompt_text_in_result") is not False
                or result.get("concurrency") != concurrency
            ):
                raise ResearchEvidenceError(f"cell identity/configuration mismatch: {stem}")
            row = next(
                item
                for item in raw["rows"]
                if item.get("tensor_parallel_size") == tp
                and item.get("concurrency") == concurrency
            )
            if any(
                row.get(field) != expected
                for field, expected in (
                    ("model_id", expected_model["hf_id"]),
                    ("model_revision", expected_model["revision"]),
                    ("workload", start["workload"]),
                    ("repetition", start["repetition"]),
                    ("input_tokens", expected_tokens[0]),
                    ("output_tokens_requested", expected_tokens[1]),
                )
            ):
                raise ResearchEvidenceError(f"raw cell identity mismatch: {stem}")
            if mode == "principal" or summary.get("failed_cells") == 0:
                _validate_successful_cell(
                    root=temporary,
                    result=result,
                    row=row,
                    stem=stem,
                    concurrency=concurrency,
                    input_tokens=expected_tokens[0],
                    output_tokens=expected_tokens[1],
                    prompt_sha256=prompt_sha256,
                )
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
            "source_commit": expected_source_commit,
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
    try:
        _check_existing_shard(repository, audit["shard_id"], destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        originals = extracted / "original"
        originals.mkdir()
        sources = (notebook, evidence_zip, runtime_path)
        if len({source.name for source in sources}) != len(sources):
            raise ResearchEvidenceError("download basenames collide in local staging")
        for source in sources:
            shutil.copy2(source, originals / source.name)
        audit["staged_at"] = str(destination)
        (extracted / "INGEST_AUDIT.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        shutil.move(str(extracted), destination)
        return audit
    except Exception:
        shutil.rmtree(extracted, ignore_errors=True)
        raise
