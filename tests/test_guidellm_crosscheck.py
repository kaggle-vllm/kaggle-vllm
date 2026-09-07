from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import kaggle_guidellm_crosscheck as runner


def args(tmp_path: Path):
    return argparse.Namespace(
        workload="balanced",
        concurrency=16,
        repetition=3,
        model_key="qwen25_3b",
        tensor_parallel_size=2,
        port=8000,
        guidellm_executable=tmp_path / "venv/bin/guidellm",
        gpu_memory_utilization=0.9,
    )


def model():
    return {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "a" * 40,
    }


def test_guidellm_command_is_exact_pinned_and_separate(tmp_path: Path):
    command = runner.guidellm_command(args(tmp_path), model(), tmp_path / "out")
    assert command[0] == str(tmp_path / "venv/bin/guidellm")
    backend = json.loads(command[command.index("--backend") + 1])
    tokenizer = json.loads(command[command.index("--tokenizer") + 1])
    assert backend["request_format"] == "/v1/completions"
    assert backend["model"] == "qwen25_3b"
    assert tokenizer["load_kwargs"]["revision"] == "a" * 40
    assert "kind=synthetic_text,prompt_tokens=512,output_tokens=256" in command[
        command.index("--data") + 1
    ]
    assert "kind=max_requests,count=48" in command
    assert "--disable-console-interactive" in command


def test_dry_run_does_not_require_guidellm_or_gpu(tmp_path: Path, capsys):
    research = tmp_path / "research"
    research.mkdir()
    (research / "model_matrix.json").write_text(
        json.dumps({"models": {"qwen25_3b": {**model(), "selected_weight_bytes": 1}}}),
        encoding="utf-8",
    )
    assert runner.main(
        [
            "--repository", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--model-key", "qwen25_3b",
            "--workload", "balanced",
            "--concurrency", "16",
            "--tensor-parallel-size", "2",
            "--guidellm-executable", str(tmp_path / "missing"),
            "--guidellm-source", str(tmp_path / "missing-source"),
            "--dry-run",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PREPARED_FOR_KAGGLE"
