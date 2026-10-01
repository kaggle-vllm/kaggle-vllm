"""Protect supplementary raw evidence and reject misleading metric changes."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_allam_evidence.py"
spec = importlib.util.spec_from_file_location("allam_evidence", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def wave():
    return audit.read_json(
        audit.EVIDENCE / "p4/allam-p4-final/runs/tp1-ctx960-online.json"
    )["waves"][0]


def test_allam_evidence_and_derived_summary():
    result = audit.validate()
    assert (
        SCRIPT.parents[1] / "research/allam/results.md"
    ).read_text() == audit.render_results(result)
    assert result == audit.read_json(SCRIPT.parents[1] / "research/allam/summary.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("wave_wall_s", 0.5),
        ("output_tokens_per_s", 0),
        ("total_output_tokens", 0),
        ("concurrency", 8),
        ("success_count", 0),
    ],
)
def test_reject_inconsistent_wave(field, value):
    record = copy.deepcopy(wave())
    record[field] = value
    with pytest.raises(AssertionError):
        audit.validate_wave(record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tpot_s", 0),
        ("ttft_s", float("nan")),
        ("completion_tokens", 0),
        ("ok", False),
        ("prompt_tokens", 127),
    ],
)
def test_reject_inconsistent_request(field, value):
    record = copy.deepcopy(wave())
    record["requests"][0][field] = value
    with pytest.raises(AssertionError):
        audit.validate_wave(record)


def test_reject_nonfinite_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"throughput": NaN}')
    with pytest.raises(ValueError, match="Non-finite"):
        audit.read_json(path)
