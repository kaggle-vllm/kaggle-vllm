"""CPU-only regression checks for the P2 Gemma cleanup and backend gates."""
from __future__ import annotations

import json
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
HELPER = runpy.run_path(str(ROOT / "scripts/fresh_kaggle_gemma_v030.py"))
RUNNER = runpy.run_path(str(ROOT / "scripts/production_acceptance_v030.py"))


class NoSuchProcess(Exception):
    pass


class AccessDenied(Exception):
    pass


PSUTIL = SimpleNamespace(STATUS_ZOMBIE="zombie", NoSuchProcess=NoSuchProcess, AccessDenied=AccessDenied)


class Child:
    def __init__(self, pid, name, status="running", start=1.0, ppid=100):
        self.pid, self._name, self._status, self._start, self._ppid = pid, name, status, start, ppid

    def oneshot(self):
        return nullcontext()

    def ppid(self):
        return self._ppid

    def status(self):
        return self._status

    def create_time(self):
        return self._start

    def name(self):
        if self._status == "zombie":
            raise NoSuchProcess("zombie name unavailable")
        return self._name

    def cmdline(self):
        if self._status == "zombie":
            raise NoSuchProcess("zombie cmdline unavailable")
        return [self._name]


class Parent:
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)

    def children(self, recursive):
        assert recursive
        return next(self.snapshots)


def llm_with_shutdown(calls, error=None):
    def shutdown(*, timeout):
        calls.append(timeout)
        if error:
            raise error
    return SimpleNamespace(upstream=SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown))))


def test_explicit_upstream_shutdown_and_normal_exit(monkeypatch):
    monkeypatch.setitem(HELPER["shutdown_and_verify"].__globals__, "gpu_process_snapshot", lambda: {"available": True, "processes": []})
    calls = []
    parent = Parent([[Child(101, "VLLM::Worker_TP0")], []])
    result = HELPER["shutdown_and_verify"](llm_with_shutdown(calls), parent, [], PSUTIL)
    assert calls == [30.0]
    assert result["status"] == "PASS"
    assert result["before_shutdown"][0]["pid"] == 101


def test_delayed_worker_exit_is_waited_for(monkeypatch):
    monkeypatch.setitem(HELPER["shutdown_and_verify"].__globals__, "gpu_process_snapshot", lambda: {"available": True, "processes": []})
    parent = Parent([[Child(101, "VLLM::Worker_TP0")], [Child(101, "VLLM::Worker_TP0")], []])
    times = iter([0, 0, 0.5, 1, 1.5])
    result = HELPER["shutdown_and_verify"](llm_with_shutdown([]), parent, [], PSUTIL, timeout=2, clock=lambda: next(times), sleep=lambda _: None)
    assert result["status"] == "PASS"
    assert result["snapshots"][0]["unresolved_pids"] == [101]


def test_zombie_does_not_count_as_live_worker(monkeypatch):
    monkeypatch.setitem(HELPER["shutdown_and_verify"].__globals__, "gpu_process_snapshot", lambda: {"available": True, "processes": []})
    result = HELPER["shutdown_and_verify"](llm_with_shutdown([]), Parent([[Child(101, "VLLM::Worker_TP0")], [Child(101, "VLLM::Worker_TP0", status="zombie")]]), [], PSUTIL)
    assert result["status"] == "PASS"
    assert result["snapshots"][0]["children"][0]["classification"] == "ZOMBIE_NOT_REAPED"


def test_pid_reuse_and_baseline_child_are_not_engine_leaks():
    classify = HELPER["classify_children"]
    before = [{"pid": 101, "start_time": 1.0}]
    after = [
        {"pid": 101, "ppid": 100, "start_time": 2.0, "status": "running", "name": "other", "cmdline": []},
        {"pid": 102, "ppid": 100, "start_time": 1.0, "status": "running", "name": "other", "cmdline": []},
    ]
    annotated, unresolved = classify(before, after, [after[1]], {"processes": []}, "zombie")
    assert [row["classification"] for row in annotated] == ["UNRELATED_PROCESS", "UNRELATED_PROCESS"]
    assert unresolved == []
    reused_worker = {**after[0], "name": "VLLM::Worker_TP0"}
    assert classify(before, [reused_worker], [], {"processes": []}, "zombie")[1] == [101]


def test_live_worker_and_unknown_owned_child_fail_bounded_check(monkeypatch):
    monkeypatch.setitem(HELPER["shutdown_and_verify"].__globals__, "gpu_process_snapshot", lambda: {"available": True, "processes": [{"pid": 101}]})
    parent = Parent([[Child(101, "VLLM::Worker_TP0")], [Child(101, "VLLM::Worker_TP0")]])
    times = iter([0, 1])
    result = HELPER["shutdown_and_verify"](llm_with_shutdown([]), parent, [], PSUTIL, timeout=1, clock=lambda: next(times))
    assert result["status"] == "FAIL"
    assert result["unresolved_pids"] == [101]
    unknown = {"pid": 103, "ppid": 100, "start_time": 1.0, "status": "running", "name": "mystery", "cmdline": []}
    assert HELPER["classify_children"]([], [unknown], [], {"processes": []}, "zombie")[1] == [103]


def test_resource_tracker_without_gpu_is_reported_but_not_tp_worker():
    tracker = {"pid": 1270, "ppid": 100, "start_time": 1.0, "status": "running", "name": "python", "cmdline": ["python", "-m", "multiprocessing.resource_tracker"]}
    annotated, unresolved = HELPER["classify_children"]([tracker], [tracker], [], {"processes": []}, "zombie")
    assert annotated[0]["classification"] == "SUPERVISOR_PROCESS"
    assert unresolved == []


def test_shutdown_exception_is_not_suppressed():
    parent = Parent([[Child(101, "VLLM::Worker_TP0")]])
    with pytest.raises(RuntimeError, match="shutdown broke"):
        HELPER["shutdown_and_verify"](llm_with_shutdown([], RuntimeError("shutdown broke")), parent, [], PSUTIL)


def test_inference_failure_preserved_and_cleanup_evidence_written(tmp_path, monkeypatch):
    import kaggle_vllm
    fake_psutil = SimpleNamespace(Process=lambda: Parent([[], []]), STATUS_ZOMBIE="zombie", NoSuchProcess=NoSuchProcess, AccessDenied=AccessDenied)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(kaggle_vllm, "activate_runtime", lambda: True)
    monkeypatch.setattr(kaggle_vllm, "KaggleSamplingParams", lambda **kwargs: kwargs)
    class BrokenLLM:
        def __init__(self, **kwargs):
            pass
        def chat(self, *args, **kwargs):
            raise ValueError("inference failed first")
    monkeypatch.setattr(kaggle_vllm, "KaggleLLM", BrokenLLM)
    monkeypatch.setitem(HELPER["main"].__globals__, "shutdown_and_verify", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cleanup also failed")))
    monkeypatch.setitem(HELPER["main"].__globals__, "gpu_process_snapshot", lambda: {"available": True, "processes": []})
    monkeypatch.setattr(sys, "argv", ["helper", "--model", str(tmp_path), "--output", str(tmp_path / "report.json")])
    with pytest.raises(ValueError, match="inference failed first"):
        HELPER["main"]()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "FAIL"
    assert "inference failed first" in report["inference_error"]
    assert "cleanup also failed" in report["cleanup_error"]


def test_backend_requires_four_actual_triton_worker_logs(tmp_path):
    runner = RUNNER["Runner"](SimpleNamespace(output_root=tmp_path))
    runner.evidence.mkdir()
    runner.logs.mkdir()
    (runner.evidence / "17_gemma.json").write_text(json.dumps({"effective_backend_diagnostic": "TRITON_ATTN"}))
    logfile = runner.logs / "17_gemma_semantic.log"
    logfile.write_text("VLLM_ATTENTION_BACKEND=TRITON_ATTN\nGemma forcing TRITON_ATTN backend\n")
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.backend()
    logfile.write_text("\n".join(f"(Worker_TP{rank} pid={100+rank}) INFO [cuda.py:478] Using AttentionBackendEnum.TRITON_ATTN backend." for rank in range(4)))
    assert runner.backend()["observed_tp_ranks"] == [0, 1, 2, 3]
    logfile.write_text(logfile.read_text() + "\n(Worker_TP0 pid=100) INFO Using AttentionBackendEnum.FLASH_ATTN backend.\n")
    with pytest.raises(RuntimeError, match="conflicting"):
        runner.backend()
