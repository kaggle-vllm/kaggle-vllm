"""Run the Milestone 1 dual-T4 benchmark matrix in isolated child processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_vllm import __version__
from kaggle_vllm.benchmark import (
    BenchmarkSpec,
    WorkloadSpec,
    build_benchmark_plan,
    prepare_evidence_directory,
    runtime_identity,
    write_json_new,
)
from kaggle_vllm.benchmark_compare import compare_results, load_benchmark_result
from kaggle_vllm.bootstrap import activate_runtime, bootstrap
from kaggle_vllm.doctor import run_doctor
from kaggle_vllm.environment import collect
from kaggle_vllm.profiles import load_profile
from kaggle_vllm.sharding import inspect_sharded_model
from kaggle_vllm.telemetry import capture_topology, run_command

CONTROL_MODEL = "facebook/opt-125m"
CONTROL_REVISION = "27dcfa74d334bc871f3234de431e71c6eeba5dd6"
QWEN_REPOSITORY = "waqasm86/kaggle-vllm-models"
QWEN_REVISION = "08bb62d0b68d20062e9009a9769c0df53d3dae21"
QWEN_SHARDS = {
    "model-rank-0-part-0.safetensors": 2_138_586_872,
    "model-rank-0-part-1.safetensors": 947_544_384,
    "model-rank-1-part-0.safetensors": 2_138_586_872,
    "model-rank-1-part-1.safetensors": 947_544_384,
}
DEFAULT_OUTPUT = Path("/kaggle/working/kaggle-vllm-tp-milestone-1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-sdk-version", default="0.2.0")
    parser.add_argument(
        "--source-identity",
        help="reviewed git commit/archive identity for the milestone source",
    )
    parser.add_argument("--control-model", default=CONTROL_MODEL)
    parser.add_argument("--control-revision", default=CONTROL_REVISION)
    parser.add_argument(
        "--qwen-model",
        type=Path,
        help="existing local TP=2 sharded_state directory; never downloaded here",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--qwen-repeats", type=int, default=3)
    parser.add_argument("--telemetry-note", default="nvidia-smi sampled by each run")
    parser.add_argument("--include-custom-all-reduce", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def benchmark_matrix(args: argparse.Namespace) -> list[tuple[str, BenchmarkSpec]]:
    control_workload = WorkloadSpec(
        max_output_tokens=128,
        warmup_runs=args.warmups,
        measurement_runs=args.repeats,
        temperature=0.0,
        ignore_eos=True,
        seed=0,
    )
    rows: list[tuple[str, BenchmarkSpec]] = []
    for tensor_parallel_size, enforce_eager in ((1, False), (2, False), (1, True), (2, True)):
        name = f"opt125m-tp{tensor_parallel_size}-eager{int(enforce_eager)}"
        rows.append(
            (
                name,
                BenchmarkSpec(
                    model=args.control_model,
                    model_revision=args.control_revision,
                    tensor_parallel_size=tensor_parallel_size,
                    max_model_len=512,
                    gpu_memory_utilization=0.40,
                    enforce_eager=enforce_eager,
                    disable_custom_all_reduce=True,
                    workload=control_workload,
                ),
            )
        )
    if args.include_custom_all_reduce:
        rows.append(
            (
                "opt125m-tp2-eager1-custom-all-reduce",
                BenchmarkSpec(
                    model=args.control_model,
                    model_revision=args.control_revision,
                    tensor_parallel_size=2,
                    max_model_len=512,
                    gpu_memory_utilization=0.40,
                    enforce_eager=True,
                    disable_custom_all_reduce=False,
                    workload=control_workload,
                ),
            )
        )

    if args.qwen_model is not None:
        qwen_workload = WorkloadSpec(
            prompts=(
                "Explain tensor parallel model sharding.",
                "What role does NCCL play in multi-GPU inference?",
                "Why can capacity improve without a throughput speedup?",
                "Describe the difference between measurement and diagnosis.",
            ),
            max_output_tokens=64,
            warmup_runs=args.warmups,
            measurement_runs=args.qwen_repeats,
            temperature=0.0,
            ignore_eos=True,
            seed=0,
        )
        common: dict[str, Any] = {
            "model": str(args.qwen_model),
            "model_revision": QWEN_REVISION,
            "model_representation": "sharded_state",
            "load_format": "sharded_state",
            "tensor_parallel_size": 2,
            "dtype": "float16",
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.70,
            "enforce_eager": True,
            "disable_custom_all_reduce": True,
            "workload": qwen_workload,
        }
        rows.extend(
            (
                ("qwen-tp2-baseline", BenchmarkSpec(**common)),
                (
                    "qwen-tp2-batched-4096",
                    BenchmarkSpec(**common, max_num_batched_tokens=4096),
                ),
            )
        )
    return rows


def validate_qwen_inspection(inspection: dict[str, Any]) -> None:
    """Require the recorded four-shard TP=2 structure without reading weights."""

    observed = {item["name"]: item["size"] for item in inspection["shards"]}
    if inspection.get("rank_count") != 2 or observed != QWEN_SHARDS:
        raise RuntimeError("Qwen artifact does not match the validated TP=2 structure")
    if inspection.get("total_size") != sum(QWEN_SHARDS.values()):
        raise RuntimeError("Qwen artifact has an unexpected total sharded size")


def _benchmark_command(spec: BenchmarkSpec, output: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "kaggle_vllm.cli",
        "benchmark",
        "--model",
        spec.model,
        "--model-representation",
        spec.model_representation,
        "--tensor-parallel-size",
        str(spec.tensor_parallel_size),
        "--dtype",
        spec.dtype,
        "--max-model-len",
        str(spec.max_model_len),
        "--gpu-memory-utilization",
        str(spec.gpu_memory_utilization),
        "--enforce-eager" if spec.enforce_eager else "--no-enforce-eager",
        (
            "--disable-custom-all-reduce"
            if spec.disable_custom_all_reduce
            else "--no-disable-custom-all-reduce"
        ),
        "--max-output-tokens",
        str(spec.workload.max_output_tokens),
        "--warmups",
        str(spec.workload.warmup_runs),
        "--repeats",
        str(spec.workload.measurement_runs),
        "--temperature",
        str(spec.workload.temperature),
        "--ignore-eos" if spec.workload.ignore_eos else "--no-ignore-eos",
        "--seed",
        str(spec.workload.seed),
        "--output",
        str(output),
    ]
    if spec.model_revision:
        command.extend(("--model-revision", spec.model_revision))
    if spec.effective_load_format:
        command.extend(("--load-format", spec.effective_load_format))
    if spec.max_num_batched_tokens is not None:
        command.extend(("--max-num-batched-tokens", str(spec.max_num_batched_tokens)))
    if spec.max_num_seqs is not None:
        command.extend(("--max-num-seqs", str(spec.max_num_seqs)))
    for prompt in spec.workload.prompts:
        command.extend(("--prompt", prompt))
    return command


def _child_environment() -> dict[str, str]:
    environment = dict(os.environ)
    source_root = Path(__file__).resolve().parents[1] / "src"
    if source_root.is_dir():
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{source_root}{os.pathsep}{existing}" if existing else str(source_root)
        )
    return environment


def _run_one(name: str, spec: BenchmarkSpec, output_dir: Path) -> Path:
    evidence = output_dir / f"{name}.json"
    log = output_dir / f"{name}.log"
    command = _benchmark_command(spec, evidence)
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=_child_environment(),
    )
    with log.open("x", encoding="utf-8") as stream:
        stream.write(result.stdout + result.stderr)
    if result.returncode != 0 or not evidence.is_file():
        write_json_new(
            output_dir / f"{name}-failure.json",
            {
                "schema_version": 1,
                "status": "failed",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "name": name,
                "returncode": result.returncode,
                "command": command,
                "log": log.name,
                "evidence_file_created": evidence.is_file(),
            },
        )
        raise RuntimeError(
            f"benchmark {name!r} failed with return code {result.returncode}; "
            f"see {log}"
        )
    load_benchmark_result(evidence)
    return evidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(output_dir: Path) -> Path:
    manifest = output_dir / "SHA256SUMS.txt"
    lines = [
        f"{_sha256(path)}  {path.name}"
        for path in sorted(output_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and path != manifest
    ]
    with manifest.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return manifest


def _comparison(
    output_dir: Path,
    filename: str,
    baseline: Path,
    candidate: Path,
) -> Path:
    payload = compare_results(
        load_benchmark_result(baseline),
        load_benchmark_result(candidate),
        baseline_label=baseline.stem,
        candidate_label=candidate.stem,
    )
    return write_json_new(output_dir / filename, payload)


def _print_command(capture: dict[str, Any]) -> None:
    print("$", " ".join(capture["command"]))
    print(capture["stdout"] or capture["stderr"] or f"[{capture['status']}]")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = benchmark_matrix(args)
    if args.dry_run:
        payload = {
            "status": "planned_not_executed",
            "output_directory": str(args.output_dir),
            "expected_sdk_version": args.expected_sdk_version,
            "qwen_model_configured": args.qwen_model is not None,
            "runs": [
                {
                    "name": name,
                    "plan": build_benchmark_plan(
                        spec, args.output_dir / f"{name}.json"
                    ),
                }
                for name, spec in matrix
            ],
            "mutations_performed": False,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if not args.source_identity:
        raise SystemExit("--source-identity is required for executed evidence")

    if __version__ != args.expected_sdk_version:
        raise SystemExit(
            f"expected kaggle-vllm {args.expected_sdk_version}, found {__version__}"
        )
    result = bootstrap(strict=True)
    activate_runtime(result.manifest)
    if run_doctor(strict=True) != 0:
        raise SystemExit("strict doctor failed; benchmark not started")

    runtime = collect()
    identity, hardware = runtime_identity(runtime)
    qwen_inspection = None
    if args.qwen_model is not None:
        inspection = inspect_sharded_model(
            args.qwen_model, expected_tensor_parallel_size=2
        )
        if not inspection.valid:
            raise SystemExit("Qwen sharded_state inspection failed")
        qwen_inspection = inspection.to_dict()
        validate_qwen_inspection(qwen_inspection)

    output_dir = prepare_evidence_directory(args.output_dir)
    profile = load_profile()
    topology = capture_topology()
    nvidia_smi = run_command(("nvidia-smi",)).to_dict()
    _print_command(nvidia_smi)
    _print_command(topology["matrix"])

    write_json_new(
        output_dir / "environment.json",
        {
            "identity": identity,
            "hardware": hardware,
            "nvidia_smi": nvidia_smi,
            "topology": topology,
        },
    )
    with (output_dir / "topology.txt").open("x", encoding="utf-8") as stream:
        stream.write(topology["matrix"]["stdout"] + "\n")
    write_json_new(
        output_dir / "run-metadata.json",
        {
            "schema_version": 1,
            "status": "execution_started",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_sdk_version": args.expected_sdk_version,
            "milestone_source_identity": args.source_identity,
            "native_runtime": {
                "source_tag": profile.vllm_source_tag,
                "source_commit": profile.vllm_source_commit,
                "wheel": profile.wheel_filename,
                "sha256": profile.wheel_sha256,
                "hf_repository": profile.hf_repo_id,
                "hf_revision": profile.hf_revision,
            },
            "control_model": {
                "repository": args.control_model,
                "revision": args.control_revision,
            },
            "qwen_model": (
                {
                    "repository": QWEN_REPOSITORY,
                    "revision": QWEN_REVISION,
                    "path": str(args.qwen_model),
                    "inspection": qwen_inspection,
                }
                if args.qwen_model is not None
                else None
            ),
            "telemetry_note": args.telemetry_note,
            "matrix": [
                {"name": name, "specification": spec.to_dict()}
                for name, spec in matrix
            ],
        },
    )

    completed: dict[str, Path] = {}
    for name, spec in matrix:
        print(f"\n=== {name} ===", flush=True)
        completed[name] = _run_one(name, spec, output_dir)

    comparisons = [
        _comparison(
            output_dir,
            "comparison-opt125m-graph-tp1-vs-tp2.json",
            completed["opt125m-tp1-eager0"],
            completed["opt125m-tp2-eager0"],
        ),
        _comparison(
            output_dir,
            "comparison-opt125m-eager-tp1-vs-tp2.json",
            completed["opt125m-tp1-eager1"],
            completed["opt125m-tp2-eager1"],
        ),
    ]
    if args.qwen_model is not None:
        comparisons.append(
            _comparison(
                output_dir,
                "comparison-qwen-tp2-batching.json",
                completed["qwen-tp2-baseline"],
                completed["qwen-tp2-batched-4096"],
            )
        )
    summary = write_json_new(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "status": "executed",
            "run_files": [path.name for path in completed.values()],
            "comparison_files": [path.name for path in comparisons],
            "qwen_characterized": args.qwen_model is not None,
            "limitations": [
                "These are offline vLLM measurements, not serving measurements.",
                "No throughput delta alone proves PCIe or NCCL causality.",
                "Results apply only to the captured workload and runtime identity.",
            ],
        },
    )
    manifest = _write_checksums(output_dir)
    print(f"\nFINAL MILESTONE 1 EVIDENCE: PASS\n{output_dir}")
    print(f"summary: {summary.name}\nchecksums: {manifest.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
