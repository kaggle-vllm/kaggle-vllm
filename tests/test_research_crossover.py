from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_vllm.research.crossover import M4_RAW_SCHEMA, analyze_m4
from kaggle_vllm.research.errors import ResearchEvidenceError


def row(
    tp: int,
    repetition: int,
    *,
    throughput: float,
    latency: float,
    concurrency: int = 16,
) -> dict:
    return {
        "schema_version": M4_RAW_SCHEMA,
        "model_id": "example/model",
        "model_revision": "a" * 40,
        "workload": "short",
        "input_tokens": 128,
        "output_tokens_requested": 64,
        "tensor_parallel_size": tp,
        "concurrency": concurrency,
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
    assert cell["tp1"]["total_tokens_per_second"]["mean_95_ci"][0] is not None
    assert cell["tp1"]["maximum_vram_mib"]["p99"] == 12000
    assert cell["tp1"]["preemptions"] is None


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


def test_verified_resource_boundary_is_counted_without_zero_throughput(
    tmp_path: Path,
) -> None:
    rows = []
    for repetition in range(5):
        rows.append(row(1, repetition, throughput=100, latency=100))
        tp2 = row(2, repetition, throughput=120, latency=90)
        if repetition < 2:
            tp2["request_failures"] = tp2["measured_requests"]
            tp2["maximum_vram_mib"] = 14_895.0
            for metric in (
                "request_throughput_per_second",
                "input_tokens_per_second",
                "output_tokens_per_second",
                "total_tokens_per_second",
                "ttft_ms",
                "tpot_ms",
                "itl_ms",
                "e2e_latency_ms",
            ):
                tp2[metric] = None
        rows.append(tp2)
    result = analyze_m4(write(tmp_path, rows))
    cell = result["cells"][0]
    assert cell["classifications"] == ["RESOURCE_BOUNDARY_OBSERVED"]
    assert cell["paired_performance_repetitions"] == 3
    assert cell["tp2_resource_boundary_repetitions"] == 2
    assert cell["tp2_over_tp1_output_speedup"] is None
    assert result["crossover_summary"][0]["resource_boundary_concurrency"] == 16


def test_over_limit_row_with_performance_value_is_not_a_resource_boundary(
    tmp_path: Path,
) -> None:
    rows = [
        row(tp, repetition, throughput=100, latency=100)
        for tp in (1, 2)
        for repetition in range(5)
    ]
    rows[0]["maximum_vram_mib"] = 14_895.0
    rows[0]["request_failures"] = 1
    with pytest.raises(ResearchEvidenceError, match="GPU memory cap"):
        analyze_m4(write(tmp_path, rows))


def test_crossover_summary_uses_first_sustained_robust_point(tmp_path: Path) -> None:
    rows = []
    for concurrency in (1, 4, 8, 16, 32, 64):
        for repetition in range(5):
            rows.extend(
                [
                    row(1, repetition, throughput=100, latency=100, concurrency=concurrency),
                    row(
                        2,
                        repetition,
                        throughput=90 if concurrency < 16 else 130,
                        latency=110 if concurrency < 16 else 80,
                        concurrency=concurrency,
                    ),
                ]
            )
    summary = analyze_m4(write(tmp_path, rows))["crossover_summary"][0]
    assert summary["principal_grid_complete"] is True
    assert summary["throughput_crossover_concurrency"] == 16
    assert summary["latency_crossover_concurrency"] == 16


def test_final_reviewed_analysis_closes_matrix_without_pseudoreplication() -> None:
    root = Path(__file__).resolve().parents[1]
    analysis = json.loads(
        (
            root
            / "artifacts/kaggle-2026-09-08-milestone-4/principal/M4_ANALYSIS.json"
        ).read_text()
    )
    assert analysis["status"] == "ANALYZED"
    assert len(analysis["cells"]) == 4 * 3 * 6
    assert len(analysis["crossover_summary"]) == 4 * 3
    assert {cell["repetitions"] for cell in analysis["cells"]} == {5}
    assert {
        cell["tp1"]["output_tokens_per_second"]["ci_unit"]
        for cell in analysis["cells"]
        if cell["tp1"]["output_tokens_per_second"] is not None
    } == {"independent_repetition_means"}

    boundaries = [
        cell
        for cell in analysis["cells"]
        if "RESOURCE_BOUNDARY_OBSERVED" in cell["classifications"]
    ]
    assert len(boundaries) == 1
    boundary = boundaries[0]
    assert boundary["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
    assert boundary["workload"] == "prefill_heavy"
    assert boundary["concurrency"] == 64
    assert boundary["tp2_resource_boundary_repetitions"] == 5
    assert boundary["paired_performance_repetitions"] == 0
    assert boundary["tp2"]["output_tokens_per_second"] is None
    assert boundary["tp2_over_tp1_output_speedup"] is None
