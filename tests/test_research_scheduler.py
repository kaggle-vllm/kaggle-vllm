from __future__ import annotations

from pathlib import Path

import pytest

from kaggle_vllm.research.errors import ResearchEvidenceError
from kaggle_vllm.research.scheduler import parse_m2_scheduler_signals


def make_logs(root: Path, *, malformed: bool = False) -> None:
    for tp in (1, 2):
        for concurrency in (1, 4, 8, 16, 32, 64):
            signal = (
                "[loggers.py:259] Running: bad reqs"
                if malformed and tp == 1 and concurrency == 1
                else "[loggers.py:259] Running: 3 reqs, Waiting: 2 reqs, GPU KV cache usage: 4.5%"
            )
            (root / f"qwen-tp{tp}-c{concurrency:02d}.server.log").write_text(signal)


def test_scheduler_signal_audit_preserves_observation_without_batch_inference(
    tmp_path: Path,
) -> None:
    make_logs(tmp_path)
    result = parse_m2_scheduler_signals(tmp_path)
    assert len(result["rows"]) == 12
    assert result["rows"][0]["running_requests"] == 3
    assert result["instantaneous_decode_batch"] == "unobserved"


def test_scheduler_signal_audit_fails_on_missing_log(tmp_path: Path) -> None:
    make_logs(tmp_path)
    (tmp_path / "qwen-tp2-c64.server.log").unlink()
    with pytest.raises(ResearchEvidenceError, match="missing logs"):
        parse_m2_scheduler_signals(tmp_path)


def test_scheduler_signal_audit_fails_on_malformed_candidate(tmp_path: Path) -> None:
    make_logs(tmp_path, malformed=True)
    with pytest.raises(ResearchEvidenceError, match="malformed required"):
        parse_m2_scheduler_signals(tmp_path)
