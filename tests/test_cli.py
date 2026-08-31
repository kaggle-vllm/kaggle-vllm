import json
from pathlib import Path

from kaggle_vllm.cli import build_parser, main


def test_cli_parses_inspect_shards():
    args = build_parser().parse_args(
        ["inspect-shards", "/model", "--json", "--tensor-parallel-size", "2"]
    )
    assert args.command == "inspect-shards"
    assert args.path == Path("/model")
    assert args.as_json
    assert args.tensor_parallel_size == 2


def test_cli_parses_dependency_doctor_options():
    args = build_parser().parse_args(
        ["doctor", "--strict", "--json", "--no-dependencies"]
    )
    assert args.strict and args.as_json
    assert args.check_dependencies is False


def test_cli_parses_verify_wheel_and_serve():
    wheel = build_parser().parse_args(
        ["verify-wheel", "vllm.whl", "--sha256", "a" * 64]
    )
    assert wheel.sha256 == "a" * 64
    server = build_parser().parse_args(
        [
            "serve",
            "/model",
            "--tensor-parallel-size",
            "2",
            "--load-format",
            "sharded_state",
            "--no-enforce-eager",
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
            "--reset-runtime",
            "--yes",
            "--staged",
            "/tmp/staged",
            "--overlay",
            "/tmp/overlay",
            "--cache",
            "/tmp/cache",
            "--manifest",
            "/tmp/runtime.json",
        ]
    )
    assert args.command == "bootstrap"
    assert args.dry_run and args.strict
    assert args.reset_runtime and args.yes
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


def test_cli_reset_dry_run_json_is_structured_and_non_mutating(tmp_path, capsys):
    staged = tmp_path / "staged"
    overlay = tmp_path / "overlay"
    cache = tmp_path / "cache"
    manifest = tmp_path / "runtime.json"

    assert (
        main(
            [
                "bootstrap",
                "--reset-runtime",
                "--dry-run",
                "--json",
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
    data = json.loads(capsys.readouterr().out)
    assert data["reset"]["safe"] is True
    assert data["reset"]["completed"] is False
    assert {
        target["label"]: target["action"] for target in data["reset"]["targets"]
    } == {
        "staged": "absent",
        "overlay": "absent",
        "manifest": "absent",
        "cache": "preserve",
    }
    assert not any(path.exists() for path in (staged, overlay, cache, manifest))


def test_cli_benchmark_dry_run_is_structured_and_non_mutating(tmp_path, capsys):
    output = tmp_path / "benchmark.json"
    assert (
        main(
            [
                "benchmark",
                "--model",
                "facebook/opt-125m",
                "--model-revision",
                "immutable",
                "--tensor-parallel-size",
                "2",
                "--max-num-batched-tokens",
                "4096",
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned_not_executed"
    assert payload["engine"]["tensor_parallel_size"] == 2
    assert payload["engine"]["max_num_batched_tokens"] == 4096
    assert not output.exists()


def test_cli_parses_compare_benchmarks():
    args = build_parser().parse_args(
        [
            "compare-benchmarks",
            "baseline.json",
            "candidate.json",
            "--candidate-label",
            "tp2",
        ]
    )
    assert args.baseline == Path("baseline.json")
    assert args.candidate == Path("candidate.json")
    assert args.candidate_label == "tp2"
