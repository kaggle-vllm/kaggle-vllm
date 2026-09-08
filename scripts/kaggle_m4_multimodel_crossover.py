#!/usr/bin/env python3
"""Run one fail-closed M4 model/workload/repetition shard on Kaggle T4 x2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_vllm import __version__
from kaggle_vllm.benchmark import (
    prepare_evidence_directory,
    runtime_identity,
    write_json_new,
)
from kaggle_vllm.doctor import run_doctor
from kaggle_vllm.environment import collect
from kaggle_vllm.exceptions import BenchmarkError
from kaggle_vllm.profiles import load_profile
from kaggle_vllm.research.crossover import M4_RAW_SCHEMA, WORKLOAD_TOKENS
from kaggle_vllm.research.errors import ResearchEvidenceError
from kaggle_vllm.research.provenance import sha256_file
from kaggle_vllm.research.resources import ResourceMonitor, require_disk_budget
from kaggle_vllm.serving_benchmark import (
    MILESTONE_CONCURRENCY,
    ServingBenchmarkSpec,
    ServingWorkloadSpec,
    build_serving_plan,
    metric_definitions,
)
from kaggle_vllm.telemetry import capture_topology, run_command

try:
    from scripts import kaggle_concurrency_crossover as lifecycle
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    import kaggle_concurrency_crossover as lifecycle  # type: ignore[no-redef]


DEFAULT_REPOSITORY = Path.cwd()
DEFAULT_OUTPUT_ROOT = Path("/kaggle/working/m4-evidence")
PROMPT_COUNT = 64
PROJECTED_EVIDENCE_BYTES = 3_000_000_000
CORPUS = (
    "Measure serving behavior under controlled tensor parallel execution. "
    "Preserve model revision workload token counts failures and timestamps. "
    "Compare throughput latency capacity memory utilization and uncertainty. "
    "Do not infer causality from topology alone or hide unsupported outcomes. "
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOAD_TOKENS), default="short")
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument(
        "--mode", choices=("compatibility", "principal", "refinement"), default="principal"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        action="append",
        help="refinement concurrency; repeat for multiple points",
    )
    parser.add_argument("--source-identity")
    parser.add_argument("--expected-sdk-version", default="0.2.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-startup-timeout", type=float, default=900.0)
    parser.add_argument("--server-shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    parser.add_argument("--vllm-executable", default="vllm")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_model(repository: Path, key: str) -> dict[str, Any]:
    path = repository / "research/model_matrix.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload["models"][key]
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise ResearchEvidenceError(f"unknown or unreadable model key {key!r}") from error
    if not isinstance(model, dict):
        raise ResearchEvidenceError(f"invalid model matrix entry {key!r}")
    return model


def _git_identity(repository: Path) -> dict[str, Any]:
    def capture(*arguments: str) -> str:
        result = subprocess.run(
            ("git", *arguments), cwd=repository, capture_output=True, text=True, check=False
        )
        if result.returncode:
            raise ResearchEvidenceError(result.stderr.strip() or "git command failed")
        return result.stdout.strip()

    status = capture("status", "--porcelain")
    return {
        "commit": capture("rev-parse", "HEAD"),
        "branch": capture("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def exact_prompt(tokenizer: Any, target_tokens: int, identity: str) -> str:
    """Find a deterministic character prefix with an exact tokenizer length."""

    nonce = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    rotated = CORPUS[len(nonce) % len(CORPUS) :] + CORPUS[: len(nonce) % len(CORPUS)]
    text = f"{nonce} {rotated}" + CORPUS * (target_tokens // 12 + 20)
    if _token_count(tokenizer, text) < target_tokens:
        raise ResearchEvidenceError("prompt corpus is unexpectedly too short")

    low, high = 1, len(text)
    while low < high:
        middle = (low + high) // 2
        if _token_count(tokenizer, text[:middle]) < target_tokens:
            low = middle + 1
        else:
            high = middle
    for end in range(max(1, low - 512), min(len(text), low + 512) + 1):
        candidate = text[:end]
        if _token_count(tokenizer, candidate) == target_tokens:
            return candidate
    raise ResearchEvidenceError(
        f"could not construct an exact {target_tokens}-token prompt for this tokenizer"
    )


def build_prompt_manifest(
    tokenizer: Any,
    *,
    model_id: str,
    revision: str,
    workload: str,
    repetition: int,
) -> dict[str, Any]:
    target, _ = WORKLOAD_TOKENS[workload]
    prompts = []
    for index in range(PROMPT_COUNT):
        identity = f"{model_id}:{revision}:{workload}:{repetition}:{index}"
        text = exact_prompt(tokenizer, target, identity)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) != target:
            raise ResearchEvidenceError("exact prompt verification failed")
        prompts.append(
            {
                "index": index,
                "text": text,
                "utf8_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "token_ids_sha256": hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "token_count": len(token_ids),
            }
        )
    if len({item["utf8_sha256"] for item in prompts}) != PROMPT_COUNT:
        raise ResearchEvidenceError("prompt generator produced duplicate prompts")
    return {
        "schema_version": "kaggle-vllm-m4-prompt-manifest-v1",
        "model_id": model_id,
        "model_revision": revision,
        "tokenizer_class": type(tokenizer).__name__,
        "token_count_method": "tokenizer.encode(add_special_tokens=True)",
        "workload": workload,
        "target_input_tokens": target,
        "repetition": repetition,
        "prefix_caching": False,
        "prompts": prompts,
    }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def matrix_specs(
    args: argparse.Namespace,
    model: Mapping[str, Any],
    prompts: Sequence[str],
    prompt_manifest_sha256: str,
) -> list[tuple[str, ServingBenchmarkSpec]]:
    input_tokens, output_tokens = WORKLOAD_TOKENS[args.workload]
    if args.mode == "compatibility":
        concurrency_values = (1,)
    elif args.mode == "refinement":
        concurrency_values = tuple(args.concurrency or ())
        if not concurrency_values or any(value < 1 or value > 64 for value in concurrency_values):
            raise ResearchEvidenceError(
                "refinement requires one or more --concurrency values in [1, 64]"
            )
    else:
        if args.concurrency:
            raise ResearchEvidenceError("--concurrency is only valid in refinement mode")
        concurrency_values = MILESTONE_CONCURRENCY
    result = []
    for concurrency in concurrency_values:
        for tp in (1, 2):
            name = f"{args.model_key}-{args.workload}-r{args.repetition:02d}-tp{tp}-c{concurrency:02d}"
            result.append(
                (
                    name,
                    ServingBenchmarkSpec(
                        model=str(model["hf_id"]),
                        model_revision=str(model["revision"]),
                        served_model_name=args.model_key,
                        tensor_parallel_size=tp,
                        base_url=f"http://127.0.0.1:{args.port}",
                        max_model_len=max(4096, input_tokens + output_tokens + 64),
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        max_num_seqs=64,
                        enable_prefix_caching=False,
                        telemetry_interval_seconds=args.telemetry_interval,
                        api_mode="completions",
                        workload=ServingWorkloadSpec(
                            concurrency=concurrency,
                            prompts=tuple(prompts),
                            prompt_profile="M4 exact tokenizer-controlled completion prompts v1",
                            max_output_tokens=output_tokens,
                            ignore_eos=True,
                            seed=args.repetition,
                            request_timeout_seconds=args.request_timeout,
                            prompt_manifest_sha256=prompt_manifest_sha256,
                            prompt_manifest_file="prompt-manifest.json",
                            retain_prompt_text_in_result=False,
                        ),
                    ),
                )
            )
    return result


def _metric_value(snapshot: Mapping[str, Any], name: str) -> list[float]:
    parsed = snapshot.get("parsed", {})
    entries = parsed.get(name, []) if isinstance(parsed, Mapping) else []
    return [float(item["value"]) for item in entries if isinstance(item, Mapping) and "value" in item]


def _resource_summary(
    samples: Sequence[Any], *, visible_gpu_indices: set[int]
) -> dict[str, float | int | None]:
    def values(attribute: str) -> list[float]:
        return [
            float(value)
            for sample in samples
            if (
                attribute.startswith("system_")
                or getattr(sample, "gpu_index", None) in visible_gpu_indices
            )
            if (value := getattr(sample, attribute, None)) is not None
        ]

    memory = values("memory_used_mib")
    utilization = values("utilization_gpu_percent")
    power = values("power_draw_w")
    temperature = values("temperature_c")
    system = values("system_used_bytes")
    return {
        "gpu_utilization_percent": statistics.fmean(utilization) if utilization else None,
        "maximum_vram_mib": max(memory) if memory else None,
        "maximum_system_ram_bytes": int(max(system)) if system else None,
        "mean_power_w": statistics.fmean(power) if power else None,
        "maximum_temperature_c": max(temperature) if temperature else None,
    }


def m4_row(
    cell: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    model: Mapping[str, Any],
    resources: Mapping[str, Any],
    prompt_manifest_sha256: str,
) -> dict[str, Any]:
    measurements = cell["measurements"]
    failed = int(measurements.get("failed_requests", 0))
    successful = int(measurements.get("successful_requests", 0))
    input_tokens, output_tokens = WORKLOAD_TOKENS[args.workload]
    if successful:
        observed_input = measurements["input_tokens_per_request"]
        observed_output = measurements["output_tokens_per_request"]
        if observed_input["min"] != input_tokens or observed_input["max"] != input_tokens:
            raise ResearchEvidenceError(
                f"server token-count mismatch: expected {input_tokens}, observed {observed_input}"
            )
        if observed_output["min"] != output_tokens or observed_output["max"] != output_tokens:
            raise ResearchEvidenceError(
                f"output-count mismatch: expected {output_tokens}, observed {observed_output}"
            )
    wall = measurements.get("measured_wall_seconds")
    available = successful > 0 and wall is not None

    metrics = cell.get("metrics", {})
    before = metrics.get("before", {}) if isinstance(metrics, Mapping) else {}
    after = metrics.get("after", {}) if isinstance(metrics, Mapping) else {}
    pre_before = _metric_value(before, "vllm:num_preemptions")
    pre_after = _metric_value(after, "vllm:num_preemptions")
    preemptions = (
        max(0, round(sum(pre_after) - sum(pre_before)))
        if pre_before or pre_after
        else None
    )
    kv_values = _metric_value(after, "vllm:kv_cache_usage_perc")

    def mean_ms(name: str) -> float | None:
        item = measurements.get(name, {})
        value = item.get("mean") if isinstance(item, Mapping) else None
        return float(value) * 1000 if value is not None else None

    return {
        "schema_version": M4_RAW_SCHEMA,
        "model_id": model["hf_id"],
        "model_revision": model["revision"],
        "workload": args.workload,
        "input_tokens": input_tokens,
        "output_tokens_requested": output_tokens,
        "tensor_parallel_size": cell["engine"]["tensor_parallel_size"],
        "concurrency": cell["concurrency"],
        "repetition": args.repetition,
        "request_throughput_per_second": measurements.get("request_throughput_per_second") if available else None,
        "input_tokens_per_second": measurements.get("input_throughput_tokens_per_second") if available else None,
        "output_tokens_per_second": measurements.get("output_throughput_tokens_per_second") if available else None,
        "total_tokens_per_second": (
            (float(measurements["input_tokens"]) + float(measurements["output_tokens"])) / float(wall)
            if available
            else None
        ),
        "ttft_ms": mean_ms("ttft_seconds") if available else None,
        "tpot_ms": mean_ms("tpot_seconds") if available else None,
        "itl_ms": mean_ms("itl_seconds") if available else None,
        "e2e_latency_ms": mean_ms("latency_seconds") if available else None,
        "request_failures": failed,
        "oom": bool(cell.get("oom_observed")),
        "warmup_requests": int(cell["workload"]["warmup_requests"]),
        "measured_requests": int(cell["workload"]["total_requests"]),
        "preemptions": preemptions,
        "kv_cache_occupancy_percent": max(kv_values) * 100 if kv_values else None,
        **resources,
        "benchmark_tool": "kaggle-vllm-m4-openai-completions-v1",
        "metric_definitions": {
            **metric_definitions(),
            "independent_unit": "fresh vLLM server process per cell and repetition",
            "token_counts": "server-reported usage must equal tokenizer-controlled request targets",
        },
        "prompt_manifest_sha256": prompt_manifest_sha256,
    }


def write_checksums(output_dir: Path) -> Path:
    return lifecycle.write_checksums(output_dir)


def prepare_shard_evidence_directory(output_root: Path, shard_name: str) -> Path:
    """Create the configured root, then create one fail-closed shard directory."""

    expanded = output_root.expanduser()
    lexical = Path(os.path.abspath(expanded))
    resolved = expanded.resolve(strict=False)
    if lexical != resolved:
        raise BenchmarkError(
            f"refusing evidence root that traverses a symlink: {lexical} -> {resolved}"
        )
    kaggle_input = Path("/kaggle/input")
    if resolved == kaggle_input or kaggle_input in resolved.parents:
        raise BenchmarkError(f"refusing to write evidence under /kaggle/input: {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise BenchmarkError(f"evidence root is not a directory: {resolved}")

    resolved.mkdir(parents=True, exist_ok=True)
    return prepare_evidence_directory(resolved / shard_name)


def _dry_run(args: argparse.Namespace, model: Mapping[str, Any]) -> int:
    placeholder = "0" * 64
    prompts = ("TOKENIZER_GENERATED_AT_EXECUTION",)
    matrix = matrix_specs(args, model, prompts, placeholder)
    print(
        json.dumps(
            {
                "status": "PREPARED_FOR_KAGGLE",
                "mode": args.mode,
                "model_key": args.model_key,
                "model_id": model["hf_id"],
                "model_revision": model["revision"],
                "workload": args.workload,
                "repetition": args.repetition,
                "exact_input_tokens": WORKLOAD_TOKENS[args.workload][0],
                "exact_output_tokens": WORKLOAD_TOKENS[args.workload][1],
                "fresh_server_per_cell": True,
                "cells": [
                    {
                        "name": name,
                        "server_command": lifecycle.server_command(
                            spec, executable=args.vllm_executable
                        ),
                        "client_plan": build_serving_plan(
                            spec, args.output_root / args.model_key / f"{name}.json"
                        ),
                    }
                    for name, spec in matrix
                ],
                "mutations_performed": False,
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.repetition < 0:
        raise SystemExit("--repetition must be non-negative")
    if args.mode == "refinement" and args.repetition < 5:
        raise SystemExit("refinement repetitions must start at 5")
    repository = args.repository.resolve()
    model = _load_model(repository, args.model_key)
    if args.dry_run:
        return _dry_run(args, model)
    if __version__ != args.expected_sdk_version:
        raise SystemExit(
            f"expected kaggle-vllm {args.expected_sdk_version}, found {__version__}"
        )
    identity = _git_identity(repository)
    if identity["dirty"]:
        raise SystemExit(f"source repository is dirty: {identity['dirty_paths']}")
    if not args.source_identity or args.source_identity != identity["commit"]:
        raise SystemExit("--source-identity must exactly equal the clean repository HEAD")
    if run_doctor(strict=True) != 0:
        raise SystemExit("strict doctor failed; M4 shard not started")

    selected_weight_bytes = int(model["selected_weight_bytes"])
    disk_guard = require_disk_budget(
        args.output_root.parent,
        projected_additional_bytes=selected_weight_bytes + PROJECTED_EVIDENCE_BYTES,
    )
    output_dir = prepare_shard_evidence_directory(
        args.output_root,
        f"{args.model_key}-{args.workload}-r{args.repetition:02d}-{args.mode}",
    )
    write_json_new(
        output_dir / "execution-start.json",
        {
            "schema_version": "kaggle-vllm-m4-execution-start-v1",
            "status": "execution_started",
            "started_at_utc": _utc_now(),
            "source": identity,
            "command": [sys.executable, *sys.argv],
            "sdk_version": __version__,
            "model_key": args.model_key,
            "model": model,
            "mode": args.mode,
            "workload": args.workload,
            "repetition": args.repetition,
            "resource_limits": {
                "per_gpu_memory_mib": 14.5 * 1024,
                "system_ram_bytes": 28 * 1024**3,
                "working_storage_bytes": 20_000_000_000,
                "disk_guard": disk_guard,
            },
            "runtime": runtime_identity(collect()),
            "native_runtime": asdict(load_profile()),
            "topology": capture_topology(),
            "nvidia_smi": run_command(("nvidia-smi",)).to_dict(),
        },
    )

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model["hf_id"],
            revision=model["revision"],
            token=os.environ.get("HF_TOKEN"),
            trust_remote_code=False,
        )
        prompt_manifest = build_prompt_manifest(
            tokenizer,
            model_id=model["hf_id"],
            revision=model["revision"],
            workload=args.workload,
            repetition=args.repetition,
        )
    except Exception as error:  # noqa: BLE001 - preserve an evidence-bearing gate failure.
        write_json_new(
            output_dir / "compatibility-gate.json",
            {
                "schema_version": "kaggle-vllm-m4-compatibility-gate-v1",
                "status": "COMPATIBILITY_GATE_FAILED",
                "stage": "pinned_tokenizer_load_or_exact_prompt_generation",
                "model_id": model["hf_id"],
                "model_revision": model["revision"],
                "error_type": type(error).__name__,
                "error_message": str(error)[:4000],
                "scientific_interpretation": "No serving compatibility conclusion; inspect access, tokenizer, and runtime evidence.",
            },
        )
        write_checksums(output_dir)
        print(f"M4 compatibility gate failed; evidence preserved in {output_dir}")
        return 2

    manifest_path = output_dir / "prompt-manifest.json"
    manifest_path.write_bytes(_canonical_json_bytes(prompt_manifest))
    manifest_sha = sha256_file(manifest_path)
    prompts = [item["text"] for item in prompt_manifest["prompts"]]
    matrix = matrix_specs(args, model, prompts, manifest_sha)
    cells: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    guard_violation: str | None = None
    semantic_gate_failure: str | None = None

    lifecycle_args = argparse.Namespace(
        vllm_executable=args.vllm_executable,
        server_startup_timeout=args.server_startup_timeout,
        server_shutdown_timeout=args.server_shutdown_timeout,
    )
    for name, spec in matrix:
        process_holder: dict[str, subprocess.Popen[Any]] = {}

        def abort_on_violation(
            _message: str, *, holder: dict[str, subprocess.Popen[Any]] = process_holder
        ) -> None:
            process = holder.get("process")
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass

        monitor = ResourceMonitor(
            phase_provider=lambda cell_name=name: (cell_name, True),
            interval_seconds=args.telemetry_interval,
            violation_callback=abort_on_violation,
        )
        monitor.start()
        try:
            cell = lifecycle._run_cell(
                name,
                spec,
                lifecycle_args,
                output_dir,
                process_observer=lambda process, holder=process_holder: holder.update(
                    process=process
                ),
            )
        finally:
            monitor.stop()
        resource_path = output_dir / f"{name}.resources.jsonl"
        with resource_path.open("x", encoding="utf-8") as stream:
            for sample in monitor.samples:
                stream.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")
        resources = _resource_summary(
            monitor.samples,
            visible_gpu_indices=(
                {0} if spec.tensor_parallel_size == 1 else {0, 1}
            ),
        )
        cells.append(cell)
        try:
            rows.append(
                m4_row(
                    cell,
                    args=args,
                    model=model,
                    resources=resources,
                    prompt_manifest_sha256=manifest_sha,
                )
            )
        except ResearchEvidenceError as error:
            semantic_gate_failure = str(error)
            write_json_new(
                output_dir / "semantic-gate-failure.json",
                {
                    "schema_version": "kaggle-vllm-m4-semantic-gate-failure-v1",
                    "status": "SEMANTIC_GATE_FAILED",
                    "cell": name,
                    "reason": semantic_gate_failure,
                    "interpretation": "Cell evidence is retained but excluded from M4 analysis.",
                },
            )
            break
        require_disk_budget(args.output_root.parent, projected_additional_bytes=0)
        if monitor.violation is not None:
            guard_violation = monitor.violation
            break

    status = (
        "RESOURCE_GUARD_VIOLATION"
        if guard_violation
        else "SEMANTIC_GATE_FAILED"
        if semantic_gate_failure
        else "executed"
    )
    result = {
        "schema_version": M4_RAW_SCHEMA,
        "status": status,
        "mode": args.mode,
        "source": identity,
        "model_key": args.model_key,
        "model_id": model["hf_id"],
        "model_revision": model["revision"],
        "workload": args.workload,
        "repetition": args.repetition,
        "prompt_manifest_sha256": manifest_sha,
        "server_lifecycle": "fresh_server_per_cell",
        "resource_guard_violation": guard_violation,
        "semantic_gate_failure": semantic_gate_failure,
        "rows": rows,
    }
    write_json_new(output_dir / "m4-raw.json", result)
    write_json_new(
        output_dir / "execution-summary.json",
        {
            "schema_version": "kaggle-vllm-m4-shard-summary-v1",
            "status": result["status"],
            "expected_cells": len(matrix),
            "completed_or_preserved_cells": len(cells),
            "failed_cells": sum(
                cell["status"] != "executed"
                or int(cell["measurements"].get("failed_requests", 0)) > 0
                for cell in cells
            ),
            "resource_guard_violation": guard_violation,
            "semantic_gate_failure": semantic_gate_failure,
            "completed_at_utc": _utc_now(),
        },
    )
    checksum_path = write_checksums(output_dir)
    print(f"M4 shard evidence: {output_dir}")
    print(f"checksums: {checksum_path}")
    return 3 if guard_violation else 4 if semantic_gate_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
