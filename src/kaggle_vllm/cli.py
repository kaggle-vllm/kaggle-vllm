"""Command-line interface for diagnostics and explicit operations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .checksums import sha256_file, verify_sha256
from .doctor import run_doctor, suggested_build_env
from .environment import as_json, collect
from .exceptions import KaggleVLLMError
from .installation import stage_wheel
from .runtime import all_sm75, all_tesla_t4, validate_tensor_parallel_size
from .server import ServerConfig, serve
from .sharding import inspect_sharded_model


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI parser (also used by CPU-only tests)."""

    parser = argparse.ArgumentParser(prog="kaggle-vllm")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="compare runtime with the validated profile")
    subcommands.add_parser("fingerprint", help="print a JSON runtime fingerprint")
    subcommands.add_parser("build-env", help="print validated source-build settings")

    verify_gpus = subcommands.add_parser("verify-gpus", help="verify T4/SM75 and TP size")
    verify_gpus.add_argument("--tensor-parallel-size", type=int, default=2)

    inspect_shards = subcommands.add_parser(
        "inspect-shards", help="inspect a persistent sharded_state directory"
    )
    inspect_shards.add_argument("path", type=Path)
    inspect_shards.add_argument("--json", action="store_true", dest="as_json")

    verify_wheel = subcommands.add_parser("verify-wheel", help="calculate/verify SHA256")
    verify_wheel.add_argument("path", type=Path)
    verify_wheel.add_argument("--sha256")

    stage = subcommands.add_parser("stage-wheel", help="stage wheel with pip --no-deps")
    stage.add_argument("path", type=Path)
    stage.add_argument("--target", type=Path, required=True)
    stage.add_argument("--sha256")

    server = subcommands.add_parser("serve", help="run upstream OpenAI-compatible server")
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return run_doctor()
        if args.command == "fingerprint":
            print(as_json())
            return 0
        if args.command == "build-env":
            for key, value in suggested_build_env().items():
                print(f"export {key}={value}")
            return 0
        if args.command == "verify-gpus":
            return _verify_gpus(args.tensor_parallel_size)
        if args.command == "inspect-shards":
            inspection = inspect_sharded_model(args.path)
            if args.as_json:
                print(json.dumps(inspection.to_dict(), indent=2))
            else:
                print(f"path: {inspection.path}")
                print(f"TP ranks: {inspection.rank_count}")
                print(f"shards: {len(inspection.shards)}")
                print(f"weight bytes: {inspection.total_size}")
                print(f"valid: {inspection.valid}")
                for warning in inspection.warnings:
                    print(f"warning: {warning}")
            return 0 if inspection.valid else 1
        if args.command == "verify-wheel":
            digest = (
                verify_sha256(args.path, args.sha256)
                if args.sha256 else sha256_file(args.path)
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
    except (KaggleVLLMError, ValueError, OSError) as error:
        parser = build_parser()
        parser.exit(2, f"kaggle-vllm: error: {error}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
