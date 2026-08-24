from pathlib import Path

from kaggle_vllm.cli import build_parser, main


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


def test_cli_parses_bootstrap_dry_run_and_path_overrides():
    args = build_parser().parse_args(
        [
            "bootstrap",
            "--dry-run",
            "--strict",
            "--staged", "/tmp/staged",
            "--overlay", "/tmp/overlay",
            "--cache", "/tmp/cache",
            "--manifest", "/tmp/runtime.json",
        ]
    )
    assert args.command == "bootstrap"
    assert args.dry_run and args.strict
    assert args.staged == Path("/tmp/staged")
    assert args.overlay == Path("/tmp/overlay")
    assert args.cache == Path("/tmp/cache")
    assert args.manifest == Path("/tmp/runtime.json")


def test_cli_bootstrap_dry_run_reports_plan_without_writes(tmp_path, capsys):
    staged = tmp_path / "staged path"
    overlay = tmp_path / "overlay"
    cache = tmp_path / "cache"
    manifest = tmp_path / "runtime.json"
    assert (
        main(
            [
                "bootstrap",
                "--dry-run",
                "--staged",
                str(staged),
                "--overlay",
                str(overlay),
                "--cache",
                str(cache),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "dry-run: no download or filesystem changes performed" in output
    assert "immutable revision: f6b4f10de54924ed6fe9e28cceab84eca7276ab6" in output
    assert "'" in next(
        line
        for line in output.splitlines()
        if "staged path" in line and "would run" in line
    )
    assert not any(path.exists() for path in (staged, overlay, cache, manifest))
