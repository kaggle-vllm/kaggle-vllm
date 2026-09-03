"""Run the Milestone 2 Qwen TP1/TP2 online concurrency matrix on Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_vllm import __version__
from kaggle_vllm.benchmark import (
    prepare_evidence_directory,
    runtime_identity,
    write_json_new,
)
from kaggle_vllm.bootstrap import activate_runtime, bootstrap
from kaggle_vllm.doctor import run_doctor
from kaggle_vllm.environment import collect
from kaggle_vllm.profiles import load_profile
from kaggle_vllm.server import ServerConfig, build_server_command
from kaggle_vllm.serving_benchmark import (
    BENCHMARK_TYPE,
    DEFAULT_MODEL,
    DEFAULT_MODEL_REVISION,
    MILESTONE_CONCURRENCY,
    SERVING_SCHEMA,
    SUMMARY_SCHEMA,
    ServingBenchmarkSpec,
    ServingWorkloadSpec,
    analyze_crossover,
    build_serving_plan,
    classify_failure,
    distribution,
    load_serving_result,
    run_serving_benchmark,
    summary_row,
    validate_serving_result,
)
from kaggle_vllm.telemetry import capture_topology, run_command

DEFAULT_OUTPUT = Path("/kaggle/working/kaggle-vllm-milestone-2")


class ServerLifecycleError(RuntimeError):
    """A server failed readiness or cleanup before useful client evidence."""

    def __init__(self, message: str, classification: str) -> None:
        super().__init__(message)
        self.classification = classification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-identity")
    parser.add_argument("--expected-sdk-version", default="0.2.0")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--model-source",
        choices=("huggingface", "local_transformers"),
        default="huggingface",
    )
    parser.add_argument("--served-model-name", default="qwen2.5-3b-instruct")
    parser.add_argument(
        "--matrix-order", choices=("interleaved", "tp-major"), default="interleaved"
    )
    parser.add_argument("--total-requests", type=int)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--server-startup-timeout", type=float, default=900.0)
    parser.add_argument("--server-shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--telemetry-interval", type=float, default=0.25)
    parser.add_argument("--vllm-executable", default="vllm")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def matrix_specs(args: argparse.Namespace) -> list[tuple[str, ServingBenchmarkSpec]]:
    """Build exactly twelve controlled cells in a recorded deterministic order."""

    pairs = (
        [(tp, concurrency) for concurrency in MILESTONE_CONCURRENCY for tp in (1, 2)]
        if args.matrix_order == "interleaved"
        else [(tp, concurrency) for tp in (1, 2) for concurrency in MILESTONE_CONCURRENCY]
    )
    rows = []
    for tp, concurrency in pairs:
        name = f"qwen-tp{tp}-c{concurrency:02d}"
        rows.append(
            (
                name,
                ServingBenchmarkSpec(
                    model=args.model,
                    model_revision=args.model_revision,
                    model_source=args.model_source,
                    served_model_name=args.served_model_name,
                    tensor_parallel_size=tp,
                    base_url=f"http://127.0.0.1:{args.port}",
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_num_batched_tokens=args.max_num_batched_tokens,
                    max_num_seqs=args.max_num_seqs,
                    telemetry_interval_seconds=args.telemetry_interval,
                    workload=ServingWorkloadSpec(
                        concurrency=concurrency,
                        total_requests=args.total_requests,
                        warmup_requests=args.warmup_requests,
                        max_output_tokens=args.max_output_tokens,
                        request_timeout_seconds=args.request_timeout,
                    ),
                ),
            )
        )
    return rows


def server_command(
    spec: ServingBenchmarkSpec, *, executable: str = "vllm"
) -> list[str]:
    """Build the pinned upstream server command; sharded_state is never selected."""

    config = ServerConfig(
        model=spec.model,
        model_revision=(
            spec.model_revision if spec.model_source == "huggingface" else None
        ),
        served_model_name=spec.served_model_name,
        tensor_parallel_size=spec.tensor_parallel_size,
        dtype=spec.dtype,
        max_model_len=spec.max_model_len,
        host="127.0.0.1",
        port=int(spec.base_url.rsplit(":", 1)[1]),
        gpu_memory_utilization=spec.gpu_memory_utilization,
        max_num_batched_tokens=spec.max_num_batched_tokens,
        max_num_seqs=spec.max_num_seqs,
        seed=spec.workload.seed,
        enforce_eager=spec.enforce_eager,
        disable_custom_all_reduce=spec.disable_custom_all_reduce,
    )
    command = build_server_command(config, executable=executable, validate_gpus=False)
    if "sharded_state" in command:
        raise ValueError("Milestone 2 refuses TP-specific sharded_state models")
    return command


def validate_local_transformers_model(path: str | Path) -> None:
    """Reject rank-specific state while accepting an attached Transformers snapshot."""

    model = Path(path)
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError("local Transformers model must be a directory with config.json")
    if any(model.glob("model-rank-*-part-*.safetensors")):
        raise ValueError("refusing topology-specific sharded_state for Milestone 2")


def wait_for_server(
    process: subprocess.Popen[Any],
    base_url: str,
    timeout: float,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Wait for /health without counting model load time as benchmark time."""

    started = clock()
    while clock() - started < timeout:
        returncode = process.poll()
        if returncode is not None:
            raise ServerLifecycleError(
                f"server exited before readiness with return code {returncode}",
                "server_start_failure",
            )
        try:
            with opener(f"{base_url.rstrip('/')}/health", timeout=2.0) as response:
                status = int(getattr(response, "status", 200))
            if status == 200:
                return {
                    "ready": True,
                    "startup_wait_seconds": clock() - started,
                    "health_status": status,
                }
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            pass
        sleeper(0.5)
    raise ServerLifecycleError(
        f"server readiness timed out after {timeout} seconds", "server_start_failure"
    )


def terminate_server(
    process: subprocess.Popen[Any], timeout: float
) -> dict[str, Any]:
    """Terminate the whole isolated server process group, escalating if needed."""

    if process.poll() is not None:
        return {"status": "already_exited", "returncode": process.returncode}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
        return {"status": "terminated", "returncode": process.returncode}
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=max(5.0, timeout))
        return {"status": "killed_after_timeout", "returncode": process.returncode}
    except (OSError, ProcessLookupError) as error:
        return {
            "status": "termination_error",
            "returncode": process.poll(),
            "error": str(error),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: Path) -> Path:
    """Write a deterministic manifest over all current regular evidence files."""

    manifest = output_dir / "SHA256SUMS.txt"
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path != manifest
    ]
    with manifest.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return manifest


def _empty_failed_cell(
    spec: ServingBenchmarkSpec,
    classification: str,
    message: str,
    *,
    server_log: str,
) -> dict[str, Any]:
    runtime = collect()
    identity, hardware = runtime_identity(runtime)
    identity["measurement_mode"] = "online_openai_chat_streaming"
    empty = distribution([])
    payload = {
        "schema_version": SERVING_SCHEMA,
        "benchmark_type": BENCHMARK_TYPE,
        "status": "failed",
        "identity": identity,
        "hardware": hardware,
        "topology": capture_topology(),
        "server": {
            "base_url": spec.base_url,
            "served_model_name": spec.served_model_name,
            "streaming": True,
            "server_log": server_log,
        },
        "engine": {
            key: value
            for key, value in spec.to_dict().items()
            if key not in {"workload", "base_url"}
        },
        "workload": spec.workload.to_dict(),
        "concurrency": spec.workload.concurrency,
        "measurements": {
            "successful_requests": 0,
            "failed_requests": int(spec.workload.total_requests or 0),
            "success_rate": 0.0,
            "failure_counts": {classification: int(spec.workload.total_requests or 0)},
            "input_tokens": 0,
            "output_tokens": 0,
            "measured_wall_seconds": None,
            "output_throughput_tokens_per_second": None,
            "ttft_seconds": empty,
            "tpot_seconds": empty,
            "latency_seconds": empty,
        },
        "gpu_telemetry": {
            "source": "nvidia-smi",
            "telemetry_sample_count": 0,
            "maximum_aggregate_sampled_memory_used_mib": None,
            "summaries": [],
            "status": "measurement_not_started",
        },
        "metrics": {"status": "measurement_not_started"},
        "failure_observations": [classification],
        "oom_observed": classification == "CUDA_OOM_observed",
        "limitations": [
            "The measured request interval did not complete; unavailable metrics are null."
        ],
        "failure_message": message[:4000],
    }
    validate_serving_result(payload)
    return payload


def _run_cell(
    name: str,
    spec: ServingBenchmarkSpec,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = output_dir / f"{name}.json"
    log_path = output_dir / f"{name}.server.log"
    command = server_command(spec, executable=args.vllm_executable)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = (
        "0" if spec.tensor_parallel_size == 1 else "0,1"
    )
    process: subprocess.Popen[Any] | None = None
    readiness: dict[str, Any] | None = None
    termination: dict[str, Any] | None = None
    failure: BaseException | None = None
    with log_path.open("x", encoding="utf-8") as server_log:
        try:
            process = subprocess.Popen(
                command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
                start_new_session=True,
            )
            readiness = wait_for_server(
                process, spec.base_url, args.server_startup_timeout
            )
            payload = run_serving_benchmark(
                spec,
                evidence,
                server_log_path=log_path,
                server_status=process.poll,
                server_metadata={
                    "server_log": log_path.name,
                    "server_command": command,
                    "visible_physical_gpu_indices": (
                        [0] if spec.tensor_parallel_size == 1 else [0, 1]
                    ),
                },
            )
        # Cell isolation requires preserving a failure JSON for unexpected client
        # or lifecycle exceptions while still allowing KeyboardInterrupt to escape.
        except Exception as error:  # noqa: BLE001
            failure = error
        finally:
            if process is not None:
                termination = terminate_server(process, args.server_shutdown_timeout)

    cleanup = run_command(
        (
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    ).to_dict()
    if failure is not None:
        log_text = ""
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        log_classification = classify_failure(log_text)
        classification = (
            "CUDA_OOM_observed"
            if log_classification == "CUDA_OOM_observed"
            else (
                failure.classification
                if isinstance(failure, ServerLifecycleError)
                else classify_failure(f"{failure}\n{log_text}")
            )
        )
        if not evidence.exists():
            payload = _empty_failed_cell(
                spec,
                classification,
                str(failure),
                server_log=log_path.name,
            )
            write_json_new(evidence, payload)
        else:
            payload = load_serving_result(evidence)
    payload["_runner_observations"] = {
        "server_command": command,
        "visible_physical_gpu_indices": (
            [0] if spec.tensor_parallel_size == 1 else [0, 1]
        ),
        "readiness": readiness,
        "termination": termination,
        "post_termination_compute_processes": cleanup,
    }
    if failure is not None:
        print(f"{name}: FAILED ({type(failure).__name__}: {failure})", flush=True)
    else:
        print(f"{name}: completed", flush=True)
    return payload


def _summary(cells: Sequence[Mapping[str, Any]], order: Sequence[str]) -> dict[str, Any]:
    model_identities = {
        (
            cell["engine"].get("model"),
            cell["engine"].get("model_revision"),
            cell["engine"].get("model_source"),
            cell["engine"].get("model_representation"),
        )
        for cell in cells
    }
    return {
        "schema_version": SUMMARY_SCHEMA,
        "benchmark_type": BENCHMARK_TYPE,
        "status": "executed_with_failures"
        if any(
            cell["status"] != "executed"
            or int(cell["measurements"].get("failed_requests", 0)) > 0
            for cell in cells
        )
        else "executed",
        "matrix_order": list(order),
        "model_identity_preserved_across_cells": len(model_identities) == 1,
        "matrix": [summary_row(cell) for cell in cells],
        "runner_observations": [
            cell.get("_runner_observations", {}) for cell in cells
        ],
        "analysis": analyze_crossover(cells),
        "interpretation_boundary": [
            "The benchmark tests whether a crossover occurs; it does not assume TP2 wins.",
            "Throughput and capacity crossovers are separate observations.",
            "No OOM is reported without matching request or server-log text.",
            "Approximately 30 GB aggregate device memory across two T4 GPUs is not one 30 GB GPU.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = matrix_specs(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema_version": SUMMARY_SCHEMA,
                    "status": "planned_not_executed",
                    "output_directory": str(args.output_dir),
                    "server_lifecycle": "fresh_server_per_cell",
                    "matrix_order": [name for name, _ in matrix],
                    "cells": [
                        {
                            "name": name,
                            "visible_physical_gpu_indices": (
                                [0] if spec.tensor_parallel_size == 1 else [0, 1]
                            ),
                            "server_command": server_command(
                                spec, executable=args.vllm_executable
                            ),
                            "client_plan": build_serving_plan(
                                spec, args.output_dir / f"{name}.json"
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

    if not args.source_identity:
        raise SystemExit("--source-identity is required for executed evidence")
    if __version__ != args.expected_sdk_version:
        raise SystemExit(
            f"expected kaggle-vllm {args.expected_sdk_version}, found {__version__}"
        )
    if args.model_source == "local_transformers":
        validate_local_transformers_model(args.model)

    bootstrap_result = bootstrap(strict=True)
    activate_runtime(bootstrap_result.manifest)
    if run_doctor(strict=True) != 0:
        raise SystemExit("strict doctor failed; benchmark not started")

    output_dir = prepare_evidence_directory(args.output_dir)
    runtime = collect()
    identity, hardware = runtime_identity(runtime)
    topology = capture_topology()
    profile = load_profile()
    write_json_new(
        output_dir / "environment.json",
        {
            "identity": identity,
            "hardware": hardware,
            "nvidia_smi": run_command(("nvidia-smi",)).to_dict(),
            "topology": topology,
        },
    )
    with (output_dir / "topology.txt").open("x", encoding="utf-8") as stream:
        stream.write(topology["matrix"]["stdout"] + "\n")
    write_json_new(
        output_dir / "run-metadata.json",
        {
            "schema_version": SUMMARY_SCHEMA,
            "status": "execution_started",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "milestone_source_identity": args.source_identity,
            "expected_sdk_version": args.expected_sdk_version,
            "server_lifecycle": "fresh_server_per_cell",
            "matrix_order_policy": args.matrix_order,
            "matrix_order": [name for name, _ in matrix],
            "model": {
                "identity": args.model,
                "revision": args.model_revision,
                "source": args.model_source,
                "representation": "transformers",
                "same_for_tp1_and_tp2": True,
            },
            "native_runtime": {
                "source_tag": profile.vllm_source_tag,
                "source_commit": profile.vllm_source_commit,
                "wheel": profile.wheel_filename,
                "sha256": profile.wheel_sha256,
                "hf_repository": profile.hf_repo_id,
                "hf_revision": profile.hf_revision,
            },
            "matrix": [
                {"name": name, "specification": spec.to_dict()}
                for name, spec in matrix
            ],
        },
    )

    cells: list[dict[str, Any]] = []
    for name, spec in matrix:
        print(f"\n=== {name} ===", flush=True)
        cells.append(_run_cell(name, spec, args, output_dir))
    summary = _summary(cells, [name for name, _ in matrix])
    write_json_new(output_dir / "summary.json", summary)
    manifest = write_checksums(output_dir)
    print(f"\nMILESTONE 2 EXECUTION COMPLETE: {output_dir}")
    print(f"checksums: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
