from pathlib import Path

from kaggle_vllm.cli import build_parser


def test_cli_parses_inspect_shards():
    args = build_parser().parse_args(["inspect-shards", "/model", "--json"])
    assert args.command == "inspect-shards"
    assert args.path == Path("/model")
    assert args.as_json


def test_cli_parses_verify_wheel_and_serve():
    wheel = build_parser().parse_args(["verify-wheel", "vllm.whl", "--sha256", "a" * 64])
    assert wheel.sha256 == "a" * 64
    server = build_parser().parse_args(
        [
            "serve", "/model", "--tensor-parallel-size", "2",
            "--load-format", "sharded_state", "--no-enforce-eager",
        ]
    )
    assert server.tensor_parallel_size == 2
    assert server.load_format == "sharded_state"
    assert server.enforce_eager is False
