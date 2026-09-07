#!/usr/bin/env python3
"""Run the Milestone 3 two-rank NCCL benchmark and build evidence."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_vllm.research.comparison import build_model_vs_observed
from kaggle_vllm.research.errors import ResearchEvidenceError
from kaggle_vllm.research.measured_comm import (
    CSV_FIELDS,
    RAW_SCHEMA,
    REQUIRED_PAYLOAD_BYTES,
    AllReduceObservation,
    fit_communication_model,
    summarize_allreduce,
)
from kaggle_vllm.research.provenance import sha256_file
from kaggle_vllm.research.report import write_m3_outputs
from kaggle_vllm.research.resources import (
    GPU_MEMORY_LIMIT_MIB,
    ResourceMonitor,
    require_disk_budget,
)
from kaggle_vllm.research.scheduler import parse_m2_scheduler_signals

EXPECTED_SDK_VERSION = "0.2.0"
EXPECTED_VLLM_SOURCE = "a26e8dc7ff2111a005144d775ecf9cebf56c45b2"
EXPECTED_NATIVE_WHEEL = (
    "vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-"
    "cp312-cp312-linux_x86_64.whl"
)
EXPECTED_NATIVE_SHA256 = (
    "5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c"
)
EXPECTED_TORCH = "2.10.0+cu128"
EXPECTED_TORCH_CUDA = "12.8"
EXPECTED_NCCL = "2.27.5"
EXPECTED_VLLM_DISTRIBUTION = "0.18.2.dev0+ga26e8dc7f.d20260822.cu128"
BENCHMARK_ID = "kaggle-vllm-m3-two-rank-nccl-allreduce-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_capture(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
        cwd=cwd,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def require_command(capture: dict[str, Any], *, label: str) -> str:
    if capture["returncode"] != 0:
        raise ResearchEvidenceError(
            f"{label} failed: {capture['stderr'] or capture['stdout']}"
        )
    return str(capture["stdout"])


def git_identity(repository: Path) -> dict[str, Any]:
    commit = require_command(
        run_capture(["git", "rev-parse", "HEAD"], cwd=repository),
        label="git rev-parse",
    )
    status = require_command(
        run_capture(["git", "status", "--porcelain"], cwd=repository),
        label="git status",
    )
    branch = require_command(
        run_capture(["git", "branch", "--show-current"], cwd=repository),
        label="git branch",
    )
    return {
        "repository": str(repository),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_paths": status.splitlines(),
    }


def validate_environment(native_wheel: Path) -> dict[str, Any]:
    import torch

    sdk_version = importlib.metadata.version("kaggle-vllm")
    vllm_version = importlib.metadata.version("vllm")
    if vllm_version != EXPECTED_VLLM_DISTRIBUTION:
        raise ResearchEvidenceError(
            f"validated vLLM distribution required: {EXPECTED_VLLM_DISTRIBUTION}; "
            f"found {vllm_version}"
        )
    if sdk_version != EXPECTED_SDK_VERSION:
        raise ResearchEvidenceError(
            f"kaggle-vllm must be {EXPECTED_SDK_VERSION}, found {sdk_version}"
        )
    if torch.__version__ != EXPECTED_TORCH or torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise ResearchEvidenceError(
            f"validated Torch/CUDA required: {EXPECTED_TORCH}/{EXPECTED_TORCH_CUDA}; "
            f"found {torch.__version__}/{torch.version.cuda}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise ResearchEvidenceError("exactly two CUDA-visible GPUs are required")
    gpus = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        total = torch.cuda.get_device_properties(index).total_memory
        if name not in {"Tesla T4", "NVIDIA Tesla T4"} or capability != [7, 5]:
            raise ResearchEvidenceError(
                f"GPU {index} is not the validated Tesla T4 SM75: {name}/{capability}"
            )
        gpus.append(
            {
                "index": index,
                "name": name,
                "compute_capability": capability,
                "total_memory_bytes": total,
                "derived_memory_guard_fraction": min(
                    0.90, GPU_MEMORY_LIMIT_MIB * 1024**2 / total
                ),
            }
        )
    nccl = ".".join(str(value) for value in torch.cuda.nccl.version())
    if nccl != EXPECTED_NCCL:
        raise ResearchEvidenceError(
            f"validated NCCL {EXPECTED_NCCL} required, found {nccl}"
        )
    if native_wheel.name != EXPECTED_NATIVE_WHEEL or not native_wheel.is_file():
        raise ResearchEvidenceError("the exact validated native wheel path is required")
    native_sha = sha256_file(native_wheel)
    if native_sha != EXPECTED_NATIVE_SHA256:
        raise ResearchEvidenceError("native wheel SHA256 does not match the baseline")
    smi = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    smi_text = require_command(smi, label="nvidia-smi identity")
    if len(smi_text.splitlines()) != 2 or any(
        "580.159.04" not in line for line in smi_text.splitlines()
    ):
        raise ResearchEvidenceError("validated NVIDIA driver 580.159.04 required")
    nvcc = run_capture(["nvcc", "--version"])
    nvcc_text = require_command(nvcc, label="nvcc identity")
    if "V12.8.93" not in nvcc_text:
        raise ResearchEvidenceError("validated CUDA toolkit 12.8.93 required")
    return {
        "schema_version": "kaggle-vllm-m3-environment-v1",
        "captured_at_utc": utc_now(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "kaggle_vllm_version": sdk_version,
        "vllm_distribution_version": vllm_version,
        "vllm_source_tag": "v0.18.1",
        "vllm_source_commit": EXPECTED_VLLM_SOURCE,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": nccl,
        "gpus": gpus,
        "nvidia_smi_identity": smi,
        "cuda_toolkit": nvcc,
        "native_wheel": native_wheel.name,
        "native_wheel_sha256": native_sha,
        "resource_limits": {
            "gpu_memory_per_device_mib": GPU_MEMORY_LIMIT_MIB,
            "system_ram_gib": 28,
            "working_storage_limit_bytes": 20_000_000_000,
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_marker(path: Path, phase: str, active: bool) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"phase": phase, "measurement_active": active}),
        encoding="utf-8",
    )
    temporary.replace(path)


def worker_main(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2 or local_rank not in {0, 1}:
        raise RuntimeError("M3 worker requires exactly two local ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    observations: list[dict[str, Any]] = []
    try:
        for payload_bytes in args.payload_bytes:
            if payload_bytes % 2:
                raise RuntimeError("FP16 payload sizes must be divisible by two")
            tensor = torch.empty(payload_bytes // 2, dtype=torch.float16, device="cuda")
            if rank == 0:
                write_marker(args.marker, f"warmup_{payload_bytes}", False)
            for _ in range(args.warmups):
                tensor.fill_(rank + 1)
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            for repetition in range(args.repetitions):
                if rank == 0:
                    write_marker(args.marker, f"measured_{payload_bytes}", True)
                local_latencies: list[float] = []
                for _ in range(args.timed_iterations):
                    tensor.fill_(rank + 1)
                    torch.cuda.synchronize()
                    dist.barrier()
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    work = dist.all_reduce(tensor, async_op=True)
                    work.wait()
                    end.record()
                    end.synchronize()
                    local_latencies.append(float(start.elapsed_time(end) * 1000))
                gathered: list[list[float] | None] | None = (
                    [None, None] if rank == 0 else None
                )
                dist.gather_object(local_latencies, gathered, dst=0)
                if rank == 0:
                    assert gathered is not None and all(value is not None for value in gathered)
                    rank0 = gathered[0]
                    rank1 = gathered[1]
                    assert rank0 is not None and rank1 is not None
                    captured_at = utc_now()
                    for iteration, (rank0_us, rank1_us) in enumerate(
                        zip(rank0, rank1, strict=True)
                    ):
                        row = AllReduceObservation(
                            schema_version=RAW_SCHEMA,
                            benchmark_id=BENCHMARK_ID,
                            payload_bytes=payload_bytes,
                            dtype="float16",
                            world_size=2,
                            repetition=args.repetition_offset + repetition,
                            iteration=iteration,
                            rank0_latency_us=rank0_us,
                            rank1_latency_us=rank1_us,
                            critical_latency_us=max(rank0_us, rank1_us),
                            warmup_iterations=args.warmups,
                            timed_iterations=args.timed_iterations,
                            captured_at_utc=captured_at,
                        )
                        observations.append(row.to_dict())
                    write_marker(args.marker, f"between_repetitions_{payload_bytes}", False)
                dist.barrier()
            expected = torch.full_like(tensor, 3.0)
            if not torch.equal(tensor, expected):
                raise RuntimeError(f"rank {rank} all-reduce correctness check failed")
            del tensor
            torch.cuda.empty_cache()
        if rank == 0:
            write_json(args.worker_output, observations)
            write_marker(args.marker, "complete", False)
    finally:
        dist.destroy_process_group()
    return 0


def parse_nccl_observations(log_path: Path) -> list[str]:
    patterns = re.compile(
        r"(NCCL version|NET/Plugin|NET/IB|NET/Socket|P2P|Channel|Trees|CollNet|NVLS|algorithm|protocol)",
        re.IGNORECASE,
    )
    selected: list[str] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if patterns.search(line) and line not in selected:
            selected.append(line[:1000])
        if len(selected) >= 200:
            break
    return selected


def read_phase(marker: Path) -> tuple[str, bool]:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        return str(payload["phase"]), bool(payload["measurement_active"])
    except (OSError, KeyError, json.JSONDecodeError):
        return "orchestration", False


def write_telemetry(path: Path, monitor: ResourceMonitor) -> None:
    fields = tuple(monitor.samples[0].to_dict())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sample.to_dict() for sample in monitor.samples)


def write_raw(output_dir: Path, records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    with (output_dir / "M3_RAW_ALLREDUCE.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    write_json(
        output_dir / "M3_RAW_ALLREDUCE.json",
        {
            "schema_version": RAW_SCHEMA,
            "benchmark_id": BENCHMARK_ID,
            "configuration": config,
            "observations": records,
        },
    )


def orchestrate(args: argparse.Namespace) -> int:
    started = utc_now()
    repository = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ResearchEvidenceError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    require_disk_budget(output_dir.parent, projected_additional_bytes=2 * 1024**3)
    environment = validate_environment(args.native_wheel.resolve())
    repo_identity = git_identity(repository)

    topology_captures = [
        run_capture(["nvidia-smi", "topo", "-m"]),
        run_capture(["nvidia-smi", "topo", "-p2p", "r"]),
        run_capture(["nvidia-smi", "topo", "-p2p", "w"]),
    ]
    topology_text = "\n\n".join(
        "COMMAND: {command}\nRETURN_CODE: {returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
            command=" ".join(capture["command"]), **capture
        )
        for capture in topology_captures
    )
    (output_dir / "topology.txt").write_text(topology_text + "\n", encoding="utf-8")
    for capture in topology_captures:
        require_command(capture, label="topology capture")
    if "PHB" not in topology_captures[0]["stdout"]:
        raise ResearchEvidenceError("primary M3 run requires observed PHB topology")

    marker = output_dir / ".measurement-phase.json"
    write_marker(marker, "startup", False)
    monitor = ResourceMonitor(phase_provider=lambda: read_phase(marker))
    child_environment = dict(os.environ)
    child_environment["NCCL_DEBUG"] = "INFO"
    child_environment.pop("NCCL_ALGO", None)
    child_environment.pop("NCCL_PROTO", None)
    log_path = output_dir / "nccl-info.log"
    commands: list[list[str]] = []
    worker_outputs: list[Path] = []
    records: list[dict[str, Any]] = []
    returncode = 0
    worker_failure: str | None = None
    with log_path.open("w", encoding="utf-8") as log:
        monitor.start()
        for repetition in range(args.repetitions):
            worker_output = output_dir / f".worker-observations-{repetition}.json"
            worker_outputs.append(worker_output)
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(Path(__file__).resolve()),
                "--worker",
                "--marker",
                str(marker),
                "--worker-output",
                str(worker_output),
                "--warmups",
                str(args.warmups),
                "--timed-iterations",
                str(args.timed_iterations),
                "--repetitions",
                "1",
                "--repetition-offset",
                str(repetition),
                "--payload-bytes",
                *[str(value) for value in args.payload_bytes],
            ]
            commands.append(command)
            log.write(f"\nM3 INDEPENDENT REPETITION {repetition}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=repository,
                env=child_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while process.poll() is None:
                time.sleep(0.25)
                if monitor.violation:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
            returncode = process.wait()
            if returncode != 0 or monitor.violation:
                break
            if not worker_output.is_file():
                worker_failure = (
                    f"rank 0 did not produce repetition {repetition} observations"
                )
                break
            try:
                repetition_records = json.loads(
                    worker_output.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                worker_failure = f"rank 0 produced malformed repetition {repetition} JSON"
                break
            if not isinstance(repetition_records, list) or not repetition_records:
                worker_failure = (
                    f"rank 0 produced empty repetition {repetition} observations"
                )
                break
            records.extend(repetition_records)
        monitor.stop()
    monitor.raise_if_violated()
    if returncode != 0:
        raise ResearchEvidenceError(
            f"two-rank benchmark failed with exit code {returncode}; inspect nccl-info.log"
        )
    if worker_failure is not None:
        raise ResearchEvidenceError(worker_failure)
    config = {
        "payload_bytes": args.payload_bytes,
        "warmup_iterations": args.warmups,
        "timed_iterations_per_repetition": args.timed_iterations,
        "independent_repetitions": args.repetitions,
        "repetition_isolation": "fresh torchrun process and NCCL initialization",
        "dtype": "float16",
        "world_size": 2,
        "timing": "per-rank CUDA events; critical path is max(rank0, rank1)",
        "correctness": "final all-reduce result equals 3.0 on both ranks",
        "nccl_debug": "INFO",
        "nccl_algo_forced": False,
        "nccl_proto_forced": False,
    }
    write_raw(output_dir, records, config)
    write_telemetry(output_dir / "gpu-telemetry.csv", monitor)
    active_samples = [sample for sample in monitor.samples if sample.measurement_active]
    if not active_samples:
        raise ResearchEvidenceError("telemetry did not sample the measured interval")
    environment["nccl_debug_observations"] = parse_nccl_observations(log_path)
    environment["telemetry"] = {
        "sample_count": len(monitor.samples),
        "measured_interval_sample_count": len(active_samples),
        "maximum_gpu_memory_used_mib": max(
            (
                sample.memory_used_mib
                for sample in monitor.samples
                if sample.memory_used_mib is not None
            ),
            default=None,
        ),
        "maximum_system_used_bytes": max(
            sample.system_used_bytes for sample in monitor.samples
        ),
        "maximum_process_tree_rss_bytes": max(
            sample.process_tree_rss_bytes for sample in monitor.samples
        ),
    }
    write_json(output_dir / "M3_ENVIRONMENT.json", environment)

    observations = tuple(AllReduceObservation(**record) for record in records)
    summaries = summarize_allreduce(observations)
    fit = fit_communication_model(summaries)
    comparison = build_model_vs_observed(
        args.m1_dir, args.m2_dir, fit=fit, payload_scenarios={}
    )
    scheduler_signals = parse_m2_scheduler_signals(args.m2_dir)
    write_json(output_dir / "M2_SCHEDULER_SIGNALS.json", scheduler_signals)
    write_m3_outputs(
        output_dir,
        summaries=summaries,
        fit=fit,
        comparison=comparison,
    )
    notebook_sha = sha256_file(args.notebook_source.resolve())
    generated_names = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )
    artifact_hashes = {
        name: sha256_file(output_dir / name) for name in generated_names
    }
    provenance = {
        "schema_version": "kaggle-vllm-m3-provenance-v1",
        "status": "MEASURED_ON_KAGGLE_PENDING_REVIEW",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "repository": repo_identity,
        "package_version": EXPECTED_SDK_VERSION,
        "source_m1": {
            "commit": "7e7355d64266e864a8113c30d52c612d98100350",
            "evidence": str(args.m1_dir),
        },
        "source_m2": {
            "accepted_main_merge": "aa2c368",
            "benchmark_commit": "4f8dcc1c032d65d54b1cce3ca213535d68fd5099",
            "evidence": str(args.m2_dir),
        },
        "native_wheel_sha256": EXPECTED_NATIVE_SHA256,
        "notebook": str(args.notebook_source),
        "notebook_sha256": notebook_sha,
        "environment": environment,
        "topology_sha256": sha256_file(output_dir / "topology.txt"),
        "generation_commands": commands,
        "configuration": config,
        "model_revisions": {
            "m1_qwen_artifact": "08bb62d0b68d20062e9009a9769c0df53d3dae21",
            "m2_qwen_transformers": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        },
        "artifact_hashes_before_provenance": artifact_hashes,
        "hash_scope_note": (
            "M3_PROVENANCE.json and SHA256SUMS.txt are excluded from the embedded "
            "hash map to avoid self-reference; SHA256SUMS authenticates provenance."
        ),
    }
    write_json(output_dir / "M3_PROVENANCE.json", provenance)
    checksum_names = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.name != "SHA256SUMS.txt"
    )
    checksum_text = "".join(
        f"{sha256_file(output_dir / name)}  {name}\n" for name in checksum_names
    )
    (output_dir / "SHA256SUMS.txt").write_text(checksum_text, encoding="utf-8")
    marker.unlink(missing_ok=True)
    for worker_output in worker_outputs:
        worker_output.unlink(missing_ok=True)
    print(json.dumps({"status": "MEASURED_ON_KAGGLE_PENDING_REVIEW", "output": str(output_dir)}, indent=2))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--native-wheel", type=Path)
    parser.add_argument("--notebook-source", type=Path)
    parser.add_argument("--m1-dir", type=Path)
    parser.add_argument("--m2-dir", type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--timed-iterations", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--repetition-offset", type=int, default=0)
    parser.add_argument(
        "--payload-bytes", type=int, nargs="+", default=list(REQUIRED_PAYLOAD_BYTES)
    )
    args = parser.parse_args(argv)
    minimum_repetitions = 1 if args.worker else 2
    if (
        args.warmups < 0
        or args.timed_iterations < 1
        or args.repetitions < minimum_repetitions
        or args.repetition_offset < 0
    ):
        parser.error(
            "warmups>=0, timed-iterations>=1, non-negative repetition offset, "
            f"and repetitions>={minimum_repetitions} are required"
        )
    if any(value <= 0 for value in args.payload_bytes):
        parser.error("payload sizes must be positive")
    if not args.worker:
        required = ("output_dir", "native_wheel", "notebook_source", "m1_dir", "m2_dir")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"orchestrator arguments missing: {missing}")
    elif args.marker is None or args.worker_output is None:
        parser.error("worker requires --marker and --worker-output")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return worker_main(args)
    created_here = args.output_dir is not None and not args.output_dir.exists()
    try:
        return orchestrate(args)
    except Exception as error:
        if created_here and args.output_dir is not None and args.output_dir.is_dir():
            write_json(
                args.output_dir / "M3_FAILURE.json",
                {
                    "schema_version": "kaggle-vllm-m3-failure-v1",
                    "status": "KAGGLE_RUN_FAILED",
                    "classification": "OBSERVED_NEGATIVE_RESULT",
                    "captured_at_utc": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error)[:4000],
                    "fabricated_measurement": False,
                },
            )
            names = sorted(
                path.name
                for path in args.output_dir.iterdir()
                if path.is_file()
                and not path.name.startswith(".")
                and path.name != "SHA256SUMS.txt"
            )
            (args.output_dir / "SHA256SUMS.txt").write_text(
                "".join(
                    f"{sha256_file(args.output_dir / name)}  {name}\n"
                    for name in names
                ),
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
