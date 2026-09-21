#!/usr/bin/env python3
"""Run one GuideLLM cross-check against a fresh canonical vLLM server."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_vllm import __version__
from kaggle_vllm.benchmark import prepare_evidence_directory, write_json_new
from kaggle_vllm.doctor import run_doctor
from kaggle_vllm.research.crossover import WORKLOAD_TOKENS
from kaggle_vllm.research.resources import ResourceMonitor, require_disk_budget
from kaggle_vllm.serving_benchmark import ServingBenchmarkSpec, ServingWorkloadSpec

try:
    from scripts import kaggle_concurrency_crossover as lifecycle
    from scripts.kaggle_m4_multimodel_crossover import _git_identity, _load_model
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    import kaggle_concurrency_crossover as lifecycle  # type: ignore[no-redef]
    from kaggle_m4_multimodel_crossover import (  # type: ignore[no-redef]
        _git_identity,
        _load_model,
    )

GUIDELLM_COMMIT = "fc2dbe9edd4f7f1a4e9ccd752f6f43591adbcb73"
GUIDELLM_DESCRIBE = "v0.7.3-47-gfc2dbe9e"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--working-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--workload", choices=tuple(WORKLOAD_TOKENS), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, choices=(1, 2), required=True)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--source-identity")
    parser.add_argument("--guidellm-executable", type=Path, required=True)
    parser.add_argument("--guidellm-source", type=Path, required=True)
    parser.add_argument("--vllm-executable", default="vllm")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--server-startup-timeout", type=float, default=900.0)
    parser.add_argument("--server-shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def guidellm_command(
    args: argparse.Namespace, model: dict[str, Any], output_dir: Path
) -> list[str]:
    input_tokens, output_tokens = WORKLOAD_TOKENS[args.workload]
    requests = max(20, args.concurrency * 3)
    backend = {
        "kind": "openai_http",
        "target": f"http://127.0.0.1:{args.port}",
        "model": args.model_key,
        "request_format": "/v1/completions",
        "stream": True,
        "timeout": 1800,
    }
    tokenizer = {
        "kind": "huggingface_auto",
        "model": model["hf_id"],
        "load_kwargs": {"revision": model["revision"], "trust_remote_code": False},
    }
    return [
        str(args.guidellm_executable),
        "run",
        "--backend",
        json.dumps(backend, separators=(",", ":")),
        "--profile",
        json.dumps(
            {
                "kind": "concurrent",
                "streams": args.concurrency,
                "max_concurrency": args.concurrency,
                "warmup": 0.1,
                "cooldown": 0.1,
            },
            separators=(",", ":"),
        ),
        "--constraint",
        f"kind=max_requests,count={requests}",
        "--constraint",
        "kind=max_errors,count=1",
        "--data",
        f"kind=synthetic_text,prompt_tokens={input_tokens},output_tokens={output_tokens},samples={requests + args.concurrency}",
        "--tokenizer",
        json.dumps(tokenizer, separators=(",", ":")),
        "--seed",
        f"kind=static,value={args.repetition}",
        "--label",
        f"model_key={args.model_key}",
        "--label",
        f"tp={args.tensor_parallel_size}",
        "--label",
        f"workload={args.workload}",
        "--label",
        f"repetition={args.repetition}",
        "--output",
        f"kind=json,path={output_dir / 'guidellm.json'}",
        "--output",
        f"kind=csv,path={output_dir / 'guidellm.csv'}",
        "--disable-console-interactive",
    ]


def server_spec(args: argparse.Namespace, model: dict[str, Any]) -> ServingBenchmarkSpec:
    input_tokens, output_tokens = WORKLOAD_TOKENS[args.workload]
    return ServingBenchmarkSpec(
        model=model["hf_id"],
        model_revision=model["revision"],
        served_model_name=args.model_key,
        tensor_parallel_size=args.tensor_parallel_size,
        base_url=f"http://127.0.0.1:{args.port}",
        max_model_len=max(4096, input_tokens + output_tokens + 64),
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=64,
        enable_prefix_caching=False,
        api_mode="completions",
        workload=ServingWorkloadSpec(
            concurrency=args.concurrency,
            max_output_tokens=output_tokens,
            seed=args.repetition,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.concurrency <= 64 or args.repetition < 0:
        raise SystemExit("concurrency must be in [1,64] and repetition non-negative")
    repository = args.repository.resolve()
    model = _load_model(repository, args.model_key)
    spec = server_spec(args, model)
    command = guidellm_command(args, model, args.output_dir)
    server_command = lifecycle.server_command(spec, executable=args.vllm_executable)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "PREPARED_FOR_KAGGLE",
                    "guidellm_commit": GUIDELLM_COMMIT,
                    "guidellm_describe": GUIDELLM_DESCRIBE,
                    "server_command": server_command,
                    "guidellm_command": command,
                    "mutations_performed": False,
                },
                indent=2,
            )
        )
        return 0
    if __version__ != "0.2.0":
        raise SystemExit(f"expected kaggle-vllm 0.2.0, found {__version__}")
    source = _git_identity(repository)
    if source["dirty"] or args.source_identity != source["commit"]:
        raise SystemExit("cross-check requires the exact clean --source-identity")
    guide_source = _git_identity(args.guidellm_source.resolve())
    if guide_source["dirty"] or guide_source["commit"] != GUIDELLM_COMMIT:
        raise SystemExit("GuideLLM source is dirty or not at the reviewed commit")
    if run_doctor(strict=True) != 0:
        raise SystemExit("strict doctor failed")
    require_disk_budget(
        args.working_root,
        projected_additional_bytes=int(model["selected_weight_bytes"]) + 3_000_000_000,
    )
    output_dir = prepare_evidence_directory(args.output_dir)
    command = guidellm_command(args, model, output_dir)
    log_path = output_dir / "server.log"
    client_log_path = output_dir / "guidellm.log"
    process_holder: dict[str, subprocess.Popen[Any]] = {}

    def abort(_message: str) -> None:
        process = process_holder.get("server")
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

    monitor = ResourceMonitor(
        phase_provider=lambda: ("guidellm_crosscheck", True),
        violation_callback=abort,
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = (
        "0" if args.tensor_parallel_size == 1 else "0,1"
    )
    client_environment = dict(os.environ)
    client_environment["CUDA_VISIBLE_DEVICES"] = ""
    server: subprocess.Popen[Any] | None = None
    readiness = None
    client_returncode = None
    termination = None
    failure: Exception | None = None
    monitor.start()
    try:
        with log_path.open("x", encoding="utf-8") as server_log:
            server = subprocess.Popen(
                server_command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
                env=environment,
                start_new_session=True,
            )
            process_holder["server"] = server
            readiness = lifecycle.wait_for_server(
                server, spec.base_url, args.server_startup_timeout
            )
            with client_log_path.open("x", encoding="utf-8") as client_log:
                client = subprocess.run(
                    command,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=client_environment,
                    check=False,
                )
                client_returncode = client.returncode
    except Exception as error:  # noqa: BLE001 - preserve an evidence-bearing failure.
        failure = error
    finally:
        if server is not None:
            termination = lifecycle.terminate_server(
                server, args.server_shutdown_timeout
            )
        monitor.stop()
    if not client_log_path.exists():
        with client_log_path.open("x", encoding="utf-8") as stream:
            if failure is not None:
                stream.write(f"{type(failure).__name__}: {failure}\n")
    with (output_dir / "resources.jsonl").open("x", encoding="utf-8") as stream:
        for sample in monitor.samples:
            stream.write(json.dumps(sample.to_dict(), sort_keys=True) + "\n")
    write_json_new(
        output_dir / "provenance.json",
        {
            "schema_version": "kaggle-vllm-guidellm-crosscheck-v1",
            "status": (
                "executed"
                if client_returncode == 0 and monitor.violation is None
                else "executed_with_failure"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "guidellm_source": guide_source,
            "guidellm_describe": GUIDELLM_DESCRIBE,
            "command": [sys.executable, *sys.argv],
            "server_command": server_command,
            "guidellm_command": command,
            "model": model,
            "workload": args.workload,
            "concurrency": args.concurrency,
            "tensor_parallel_size": args.tensor_parallel_size,
            "repetition": args.repetition,
            "readiness": readiness,
            "client_returncode": client_returncode,
            "failure_type": type(failure).__name__ if failure is not None else None,
            "failure_message": str(failure)[:4000] if failure is not None else None,
            "termination": termination,
            "resource_guard_violation": monitor.violation,
            "metric_boundary": (
                "GuideLLM metrics retain GuideLLM definitions and are not silently "
                "equated with kaggle-vllm client metrics."
            ),
        },
    )
    lifecycle.write_checksums(output_dir)
    return (
        0
        if client_returncode == 0 and monitor.violation is None and failure is None
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
