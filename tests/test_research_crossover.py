from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_vllm.research.crossover import M4_RAW_SCHEMA, analyze_m4
from kaggle_vllm.research.errors import ResearchEvidenceError


def row(tp: int, repetition: int, *, throughput: float, latency: float) -> dict:
    return {
        "schema_version": M4_RAW_SCHEMA,
        "model_id": "example/model",
        "model_revision": "a" * 40,
        "workload": "short",
        "input_tokens": 128,
        "output_tokens_requested": 64,
        "tensor_parallel_size": tp,
        "concurrency": 16,
        "repetition": repetition,
        "request_throughput_per_second": throughput / 64,
        "input_tokens_per_second": throughput * 2,
        "output_tokens_per_second": throughput,
        "total_tokens_per_second": throughput * 3,
        "ttft_ms": latency / 2,
        "tpot_ms": latency / 64,
        "itl_ms": latency / 65,
        "e2e_latency_ms": latency,
        "request_failures": 0,
        "oom": False,
        "warmup_requests": 4,
        "measured_requests": 32,
        "preemptions": None,
        "kv_cache_occupancy_percent": 10.0,
        "gpu_utilization_percent": 80.0,
        "maximum_vram_mib": 12000.0,
        "maximum_system_ram_bytes": 20 * 1024**3,
        "mean_power_w": 65.0,
        "maximum_temperature_c": 70.0,
        "benchmark_tool": "kaggle-vllm-m2-compatible",
        "metric_definitions": {"ttft_ms": "request start to first streamed token"},
        "prompt_manifest_sha256": "0" * 64,
    }


def write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "m4.json"
    path.write_text(json.dumps({"schema_version": M4_RAW_SCHEMA, "rows": rows}))
    return path


def test_robust_throughput_and_latency_crossovers(tmp_path: Path) -> None:
    rows = []
    for repetition in range(5):
        rows.extend(
            [
                row(1, repetition, throughput=100 + repetition, latency=100 + repetition),
                row(2, repetition, throughput=130 + repetition, latency=80 + repetition),
            ]
        )
    cell = analyze_m4(write(tmp_path, rows))["cells"][0]
    assert cell["classifications"] == [
        "THROUGHPUT_CROSSOVER",
        "LATENCY_CROSSOVER",
    ]
    assert cell["tp2_over_tp1_output_speedup"]["mean"] > 1
    assert cell["tp1"]["ttft_ms"]["independent_count"] == 5


def test_incomplete_repetitions_fail_closed(tmp_path: Path) -> None:
    rows = [row(tp, repetition, throughput=100, latency=100) for tp in (1, 2) for repetition in range(4)]
    with pytest.raises(ResearchEvidenceError, match="matched independent"):
        analyze_m4(write(tmp_path, rows))


@pytest.mark.parametrize("mutation", ["nan", "tokens", "duplicate", "missing"])
def test_adversarial_rows_fail_closed(tmp_path: Path, mutation: str) -> None:
    rows = [row(tp, repetition, throughput=100, latency=100) for tp in (1, 2) for repetition in range(5)]
    if mutation == "nan":
        rows[0]["ttft_ms"] = float("nan")
    elif mutation == "tokens":
        rows[0]["input_tokens"] = 127
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0].pop("itl_ms")
    with pytest.raises(ResearchEvidenceError):
        analyze_m4(write(tmp_path, rows))


def test_capacity_is_not_mislabeled_as_throughput(tmp_path: Path) -> None:
    rows = []
    for repetition in range(5):
        tp1 = row(1, repetition, throughput=1, latency=1000)
        tp1["oom"] = True
        tp1["request_failures"] = 1
        for metric in (
            "request_throughput_per_second", "input_tokens_per_second",
            "output_tokens_per_second", "total_tokens_per_second", "ttft_ms",
            "tpot_ms", "itl_ms", "e2e_latency_ms",
        ):
            tp1[metric] = None
        rows.extend([tp1, row(2, repetition, throughput=100, latency=100)])
    cell = analyze_m4(write(tmp_path, rows))["cells"][0]
    assert cell["classifications"] == ["CAPACITY_CROSSOVER"]
    assert cell["tp2_over_tp1_output_speedup"] is None
