from __future__ import annotations

import hashlib
import json
import signal
from types import SimpleNamespace

import pytest

from scripts import kaggle_concurrency_crossover as runner


def test_exact_tp1_tp2_concurrency_matrix_and_model_identity():
    args = runner.build_parser().parse_args(["--dry-run"])
    matrix = runner.matrix_specs(args)
    assert len(matrix) == 12
    assert [spec.workload.concurrency for _, spec in matrix] == [
        1,
        1,
        4,
        4,
        8,
        8,
        16,
        16,
        32,
        32,
        64,
        64,
    ]
    assert [spec.tensor_parallel_size for _, spec in matrix] == [1, 2] * 6
    identities = {(spec.model, spec.model_revision) for _, spec in matrix}
    assert len(identities) == 1


def test_tp_major_order_and_server_command_have_expected_tp():
    args = runner.build_parser().parse_args(["--dry-run", "--matrix-order", "tp-major"])
    matrix = runner.matrix_specs(args)
    assert [spec.tensor_parallel_size for _, spec in matrix] == [1] * 6 + [2] * 6
    command = runner.server_command(matrix[-1][1])
    index = command.index("--tensor-parallel-size")
    assert command[index + 1] == "2"
    assert command[command.index("--revision") + 1] == args.model_revision
    assert "sharded_state" not in command
    assert "--no-enable-prefix-caching" in command


def test_local_tp_specific_sharded_state_is_rejected(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model-rank-0-part-0.safetensors").write_bytes(b"not weights")
    with pytest.raises(ValueError, match="sharded_state"):
        runner.validate_local_transformers_model(model)


class FakeProcess:
    def __init__(self, polls):
        self.polls = iter(polls)
        self.pid = 123
        self.returncode = None

    def poll(self):
        try:
            value = next(self.polls)
        except StopIteration:
            value = self.returncode
        if value is not None:
            self.returncode = value
        return value

    def wait(self, timeout):
        self.returncode = -signal.SIGTERM
        return self.returncode


def test_server_startup_exit_and_timeout_are_classified():
    with pytest.raises(runner.ServerLifecycleError, match="exited") as exited:
        runner.wait_for_server(FakeProcess([7]), "http://127.0.0.1:8000", 5)
    assert exited.value.classification == "server_start_failure"

    times = iter((0.0, 0.1, 1.1))
    with pytest.raises(runner.ServerLifecycleError, match="timed out"):
        runner.wait_for_server(
            FakeProcess([None, None]),
            "http://127.0.0.1:8000",
            1,
            opener=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no")),
            clock=lambda: next(times),
            sleeper=lambda _seconds: None,
        )


def test_clean_server_termination_targets_process_group(monkeypatch):
    process = FakeProcess([None])
    calls = []
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    result = runner.terminate_server(process, 3)
    assert result["status"] == "terminated"
    assert calls == [(123, signal.SIGTERM)]


def test_failed_cell_payload_still_validates_and_records_null_measurements(monkeypatch):
    monkeypatch.setattr(runner, "collect", lambda: SimpleNamespace())
    monkeypatch.setattr(
        runner,
        "runtime_identity",
        lambda _runtime: ({"measurement_mode": "test"}, {}),
    )
    monkeypatch.setattr(runner, "capture_topology", dict)
    args = runner.build_parser().parse_args(["--dry-run"])
    spec = runner.matrix_specs(args)[0][1]
    payload = runner._empty_failed_cell(
        spec,
        "server_start_failure",
        "server never became ready",
        server_log="qwen-tp1-c01.server.log",
    )
    assert payload["status"] == "failed"
    assert payload["measurements"]["ttft_seconds"]["p95"] is None
    assert payload["measurements"]["output_throughput_tokens_per_second"] is None
    assert payload["failure_observations"] == ["server_start_failure"]

    oom_payload = runner._empty_failed_cell(
        spec,
        "server_start_failure",
        "server never became ready",
        server_log="qwen-tp1-c01.server.log",
        oom_observed=True,
    )
    assert oom_payload["failure_observations"] == [
        "server_start_failure",
        "CUDA_OOM_observed",
    ]
    assert oom_payload["oom_observed"] is True


def test_partial_server_start_failure_writes_all_cell_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess([7, 7]))
    monkeypatch.setattr(runner, "collect", lambda: SimpleNamespace())
    monkeypatch.setattr(
        runner,
        "runtime_identity",
        lambda _runtime: ({"measurement_mode": "test"}, {}),
    )
    monkeypatch.setattr(runner, "capture_topology", dict)
    monkeypatch.setattr(
        runner,
        "run_command",
        lambda _command: SimpleNamespace(
            to_dict=lambda: {"status": "unavailable", "stdout": "", "stderr": ""}
        ),
    )
    args = runner.build_parser().parse_args(["--dry-run"])
    name, specification = runner.matrix_specs(args)[0]
    payload = runner._run_cell(name, specification, args, tmp_path)
    assert payload["status"] == "failed"
    assert payload["failure_observations"] == ["server_start_failure"]
    assert (tmp_path / f"{name}.json").is_file()
    assert (tmp_path / f"{name}.server.log").is_file()
    assert (tmp_path / f"{name}.metrics.txt").is_file()
    assert (tmp_path / f"{name}.telemetry.jsonl").is_file()
    assert (tmp_path / f"{name}-requests.jsonl").is_file()


def test_checksum_manifest_is_name_sorted_and_deterministic(tmp_path):
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    manifest = runner.write_checksums(tmp_path)
    lines = manifest.read_text(encoding="utf-8").splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == ["a.txt", "z.txt"]
    assert lines[0].startswith(hashlib.sha256(b"a").hexdigest())


def test_matrix_dry_run_writes_nothing_and_prints_all_cells(tmp_path, capsys):
    output = tmp_path / "evidence"
    assert runner.main(["--dry-run", "--output-dir", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned_not_executed"
    assert len(payload["cells"]) == 12
    assert payload["server_lifecycle"] == "fresh_server_per_cell"
    assert not output.exists()
