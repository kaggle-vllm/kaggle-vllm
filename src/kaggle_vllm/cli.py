"""Command-line interface for diagnostics and explicit operations."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Sequence
from pathlib import Path

from .benchmark import (
    BenchmarkSpec,
    WorkloadSpec,
    build_benchmark_plan,
    run_offline_benchmark,
    write_json_new,
)
from .benchmark_compare import compare_results, load_benchmark_result
from .bootstrap import (
    DEFAULT_CACHE,
    DEFAULT_MANIFEST,
    DEFAULT_OVERLAY,
    DEFAULT_STAGED,
    shell_exports,
)
from .bootstrap import bootstrap as bootstrap_runtime
from .checksums import sha256_file, verify_sha256
from .doctor import run_doctor, suggested_build_env
from .environment import as_json, collect
from .exceptions import KaggleVLLMError
from .installation import stage_wheel
from .profiles import DEFAULT_PROFILE
from .runtime import all_sm75, all_tesla_t4, validate_tensor_parallel_size
from .server import ServerConfig, serve
from .sharding import inspect_sharded_model


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser (also used by CPU-only tests)."""

    parser = argparse.ArgumentParser(prog="kaggle-vllm")
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser(
        "doctor", help="compare runtime and dependencies with the validated profile"
    )
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.add_argument(
        "--no-dependencies",
        action="store_false",
        dest="check_dependencies",
        help="skip installed-distribution checks",
    )
    subcommands.add_parser("fingerprint", help="print a JSON runtime fingerprint")
    subcommands.add_parser("build-env", help="print validated source-build settings")

    bootstrap = subcommands.add_parser(
        "bootstrap",
        help="explicitly download and stage the validated Kaggle native runtime",
    )
    bootstrap.add_argument("--profile", default=DEFAULT_PROFILE)
    bootstrap.add_argument("--staged", type=Path, default=DEFAULT_STAGED)
    bootstrap.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    bootstrap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    bootstrap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    bootstrap.add_argument("--strict", action="store_true")
    bootstrap.add_argument("--dry-run", action="store_true")
    bootstrap.add_argument(
        "--reset-runtime",
        action="store_true",
        help="reset validated SDK-owned runtime state before bootstrapping",
    )
    bootstrap.add_argument(
        "--yes",
        action="store_true",
        help="confirm the destructive reset plan (requires --reset-runtime)",
    )
    bootstrap.add_argument("--json", action="store_true", dest="as_json")

    runtime_env = subcommands.add_parser(
        "env", help="print activation exports from a completed bootstrap manifest"
    )
    runtime_env.add_argument("--manifest", type=Path, default=None)

    verify_gpus = subcommands.add_parser(
        "verify-gpus", help="verify T4/SM75 and TP size"
    )
    verify_gpus.add_argument("--tensor-parallel-size", type=int, default=2)

    inspect_shards = subcommands.add_parser(
        "inspect-shards", help="inspect a persistent sharded_state directory"
    )
    inspect_shards.add_argument("path", type=Path)
    inspect_shards.add_argument("--json", action="store_true", dest="as_json")
    inspect_shards.add_argument("--tensor-parallel-size", type=int)

    verify_wheel = subcommands.add_parser(
        "verify-wheel", help="calculate/verify SHA256"
    )
    verify_wheel.add_argument("path", type=Path)
    verify_wheel.add_argument("--sha256")

    stage = subcommands.add_parser("stage-wheel", help="stage wheel with pip --no-deps")
    stage.add_argument("path", type=Path)
    stage.add_argument("--target", type=Path, required=True)
    stage.add_argument("--sha256")

    server = subcommands.add_parser(
        "serve", help="run upstream OpenAI-compatible server"
    )
    server.add_argument("model")
    server.add_argument("--served-model-name")
    server.add_argument("--tensor-parallel-size", type=int, default=1)
    server.add_argument("--load-format")
    server.add_argument("--dtype", default="float16")
    server.add_argument("--max-model-len", type=int)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--gpu-memory-utilization", type=float)
    server.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    server.add_argument(
        "--disable-custom-all-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    server.add_argument("--vllm-executable", default="vllm")

    benchmark = subcommands.add_parser(
        "benchmark", help="measure one upstream vLLM offline-engine configuration"
    )
    benchmark.add_argument("--model", required=True)
    benchmark.add_argument("--model-revision")
    benchmark.add_argument(
        "--model-representation",
        choices=("transformers", "sharded_state"),
        default="transformers",
    )
    benchmark.add_argument("--load-format")
    benchmark.add_argument("--tensor-parallel-size", type=int, required=True)
    benchmark.add_argument("--dtype", default="float16")
    benchmark.add_argument("--max-model-len", type=int, default=512)
    benchmark.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    benchmark.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    benchmark.add_argument(
        "--disable-custom-all-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    benchmark.add_argument("--max-num-batched-tokens", type=int)
    benchmark.add_argument("--max-num-seqs", type=int)
    benchmark.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="repeat for a deterministic multi-request batch",
    )
    benchmark.add_argument("--max-output-tokens", type=int, default=128)
    benchmark.add_argument("--warmups", type=int, default=1)
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--temperature", type=float, default=0.0)
    benchmark.add_argument(
        "--ignore-eos", action=argparse.BooleanOptionalAction, default=True
    )
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--dry-run", action="store_true")

    compare = subcommands.add_parser(
        "compare-benchmarks", help="compare two schema-v1 benchmark JSON files"
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--baseline-label", default="baseline")
    compare.add_argument("--candidate-label", default="candidate")
    compare.add_argument("--output", type=Path)
    return parser


def _verify_gpus(tensor_parallel_size: int) -> int:
    environment = collect()
    validate_tensor_parallel_size(tensor_parallel_size, environment.gpu_count)
    if not all_tesla_t4(environment.gpus) or not all_sm75(environment.gpus):
        raise KaggleVLLMError("visible GPUs do not all match Tesla T4 / SM75")
    print(
        f"PASS: {environment.gpu_count} visible Tesla T4 / SM75 GPUs support "
        f"tensor_parallel_size={tensor_parallel_size}"
    )
    return 0


def _run_bootstrap(args: argparse.Namespace) -> int:
    result = bootstrap_runtime(
        profile_name=args.profile,
        staged=args.staged,
        overlay=args.overlay,
        cache=args.cache,
        manifest=args.manifest,
        strict=args.strict,
        dry_run=args.dry_run,
        reset_runtime=args.reset_runtime,
        yes=args.yes,
    )
    data = result.to_dict()
    if args.as_json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"profile: {data['profile']}")
    print(f"compatible: {data['compatible']}")
    print(f"strict: {data['strict']}")
    print(f"HF repository: {data['artifact']['hf_repo_id']}")
    print(f"immutable revision: {data['artifact']['hf_revision']}")
    print(f"wheel: {data['artifact']['filename']}")
    print(f"expected SHA256: {data['artifact']['sha256']}")
    print(f"cache: {data['paths']['cache']}")
    print(f"staged: {data['paths']['staged']}")
    print(f"overlay: {data['paths']['overlay']}")
    print(f"manifest: {data['paths']['manifest']}")
    if data["reset"]:
        print(f"reset plan safe: {data['reset']['safe']}")
        for target in data["reset"]["targets"]:
            print(
                f"reset {target['label']}: {target['action']} {target['path']} "
                f"(exists={target['exists']}, owned={target['owned']}; "
                f"{target['reason']})"
            )
    for finding in data["findings"]:
        print(f"{finding['status'].upper()}: {finding['message']}")
    if args.dry_run:
        print("dry-run: no download or filesystem changes performed")
        for command in data["commands"]:
            print("would run: " + shlex.join(command))
    elif result.already_complete:
        print("bootstrap already complete; matching manifest reused")
    else:
        if result.reset_completed:
            print(f"runtime reset complete; cache preserved: {result.plan.paths.cache}")
        print(f"bootstrap complete: {result.manifest}")
    return 0


def _benchmark_spec(args: argparse.Namespace) -> BenchmarkSpec:
    workload_options = {
        "max_output_tokens": args.max_output_tokens,
        "warmup_runs": args.warmups,
        "measurement_runs": args.repeats,
        "temperature": args.temperature,
        "ignore_eos": args.ignore_eos,
        "seed": args.seed,
    }
    if args.prompts:
        workload_options["prompts"] = tuple(args.prompts)
    workload = WorkloadSpec(**workload_options)
    return BenchmarkSpec(
        model=args.model,
        model_revision=args.model_revision,
        model_representation=args.model_representation,
        load_format=args.load_format,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        workload=workload,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return run_doctor(
                strict=args.strict,
                check_dependencies=args.check_dependencies,
                as_json_output=args.as_json,
            )
        if args.command == "fingerprint":
            print(as_json())
            return 0
        if args.command == "build-env":
            for key, value in suggested_build_env().items():
                print(f"export {key}={value}")
            return 0
        if args.command == "bootstrap":
            return _run_bootstrap(args)
        if args.command == "env":
            for line in shell_exports(args.manifest):
                print(line)
            return 0
        if args.command == "verify-gpus":
            return _verify_gpus(args.tensor_parallel_size)
        if args.command == "inspect-shards":
            inspection = inspect_sharded_model(
                args.path,
                expected_tensor_parallel_size=args.tensor_parallel_size,
            )
            if args.as_json:
                print(json.dumps(inspection.to_dict(), indent=2))
            else:
                print(f"path: {inspection.path}")
                print(f"TP ranks: {inspection.rank_count}")
                print(f"shards: {len(inspection.shards)}")
                print(f"weight bytes: {inspection.total_size}")
                print(f"valid: {inspection.valid}")
                for error in inspection.topology_errors:
                    print(f"error: {error}")
                for warning in inspection.warnings:
                    print(f"warning: {warning}")
            return 0 if inspection.valid else 1
        if args.command == "verify-wheel":
            digest = (
                verify_sha256(args.path, args.sha256)
                if args.sha256
                else sha256_file(args.path)
            )
            print(f"{digest}  {args.path.name}")
            return 0
        if args.command == "stage-wheel":
            target = stage_wheel(args.path, args.target, expected_sha256=args.sha256)
            print(f"staged without dependencies: {target}")
            return 0
        if args.command == "serve":
            config = ServerConfig(
                model=args.model,
                served_model_name=args.served_model_name,
                tensor_parallel_size=args.tensor_parallel_size,
                load_format=args.load_format,
                dtype=args.dtype,
                max_model_len=args.max_model_len,
                host=args.host,
                port=args.port,
                gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                disable_custom_all_reduce=args.disable_custom_all_reduce,
            )
            return serve(config, executable=args.vllm_executable)
        if args.command == "benchmark":
            specification = _benchmark_spec(args)
            payload = (
                build_benchmark_plan(specification, args.output)
                if args.dry_run
                else run_offline_benchmark(specification, args.output)
            )
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "compare-benchmarks":
            payload = compare_results(
                load_benchmark_result(args.baseline),
                load_benchmark_result(args.candidate),
                baseline_label=args.baseline_label,
                candidate_label=args.candidate_label,
            )
            if args.output:
                write_json_new(args.output, payload)
            print(json.dumps(payload, indent=2))
            return 0
    except (KaggleVLLMError, ValueError, OSError) as error:
        parser = build_parser()
        parser.exit(2, f"kaggle-vllm: error: {error}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
