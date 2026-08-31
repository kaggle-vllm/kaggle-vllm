"""Validation and conservative comparison of benchmark evidence."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .benchmark import MEASUREMENT_MODE, SCHEMA_VERSION
from .exceptions import BenchmarkError


def percentage_delta(baseline: float | None, candidate: float | None) -> float | None:
    """Calculate ``(candidate - baseline) / baseline * 100`` when defined."""

    if baseline is None or candidate is None:
        return None
    baseline_value = float(baseline)
    candidate_value = float(candidate)
    if not math.isfinite(baseline_value) or not math.isfinite(candidate_value):
        return None
    if baseline_value == 0:
        return None
    return (candidate_value - baseline_value) / baseline_value * 100.0


def validate_benchmark_result(payload: Mapping[str, Any]) -> None:
    """Validate the stable structural contract consumed by comparisons."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkError("unsupported benchmark schema_version")
    if payload.get("status") != "executed":
        raise BenchmarkError("benchmark result status must be 'executed'")
    for section in (
        "identity",
        "hardware",
        "topology",
        "engine",
        "workload",
        "measurements",
        "gpu_telemetry",
    ):
        if not isinstance(payload.get(section), Mapping):
            raise BenchmarkError(f"benchmark result is missing {section!r}")
    if not isinstance(payload.get("limitations"), list):
        raise BenchmarkError("benchmark result is missing 'limitations'")
    if payload["identity"].get("measurement_mode") != MEASUREMENT_MODE:
        raise BenchmarkError("benchmark measurement mode is not supported")
    aggregate = payload["measurements"].get("aggregate", {})
    for metric in (
        "aggregate_input_tokens_per_second",
        "aggregate_output_tokens_per_second",
        "total_wall_duration_seconds",
        "requests_completed",
    ):
        if metric not in aggregate:
            raise BenchmarkError(f"benchmark aggregate is missing {metric!r}")


def load_benchmark_result(path: str | Path) -> dict[str, Any]:
    """Load and validate one benchmark evidence JSON file."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"unable to read benchmark result {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BenchmarkError("benchmark result must be a JSON object")
    validate_benchmark_result(payload)
    return payload


def _nested(data: Mapping[str, Any], *path: str) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _mean_peak_memory(result: Mapping[str, Any]) -> float | None:
    summaries = _nested(result, "gpu_telemetry", "summaries") or []
    values = [
        float(item["peak_memory_used_mib"])
        for item in summaries
        if item.get("peak_memory_used_mib") is not None
    ]
    return statistics.fmean(values) if values else None


def _active_gpu_indices(result: Mapping[str, Any]) -> list[int]:
    summaries = _nested(result, "gpu_telemetry", "summaries") or []
    return [
        int(item["index"])
        for item in summaries
        if (item.get("peak_utilization_percent") or 0) > 0
    ]


def compare_results(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    """Compare two compatible evidence files without assigning unsupported causes."""

    validate_benchmark_result(baseline)
    validate_benchmark_result(candidate)
    baseline_aggregate = baseline["measurements"]["aggregate"]
    candidate_aggregate = candidate["measurements"]["aggregate"]
    baseline_memory = _mean_peak_memory(baseline)
    candidate_memory = _mean_peak_memory(candidate)

    comparison_fields = (
        "model",
        "model_revision",
        "model_representation",
        "load_format",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "enforce_eager",
        "disable_custom_all_reduce",
        "max_num_batched_tokens",
        "max_num_seqs",
    )
    differing_engine_fields = [
        field_name
        for field_name in comparison_fields
        if baseline["engine"].get(field_name) != candidate["engine"].get(field_name)
    ]
    workload_equal = baseline["workload"] == candidate["workload"]
    baseline_tp = baseline["engine"].get("tensor_parallel_size")
    candidate_tp = candidate["engine"].get("tensor_parallel_size")
    controlled_tp = (
        workload_equal
        and not differing_engine_fields
        and baseline_tp != candidate_tp
    )

    throughput_delta = percentage_delta(
        baseline_aggregate["aggregate_output_tokens_per_second"],
        candidate_aggregate["aggregate_output_tokens_per_second"],
    )
    latency_delta = percentage_delta(
        baseline_aggregate["trial_wall_duration_seconds"]["mean"],
        candidate_aggregate["trial_wall_duration_seconds"]["mean"],
    )
    load_delta = percentage_delta(
        baseline["measurements"].get("model_load_wall_seconds"),
        candidate["measurements"].get("model_load_wall_seconds"),
    )
    memory_delta = (
        candidate_memory - baseline_memory
        if baseline_memory is not None and candidate_memory is not None
        else None
    )

    observations: list[str] = []
    if throughput_delta is not None:
        direction = "higher" if throughput_delta >= 0 else "lower"
        observations.append(
            f"{candidate_label} output throughput was "
            f"{abs(throughput_delta):.2f}% {direction} than {baseline_label}."
        )
    if latency_delta is not None:
        direction = "higher" if latency_delta >= 0 else "lower"
        observations.append(
            f"{candidate_label} mean batch latency was "
            f"{abs(latency_delta):.2f}% {direction} than {baseline_label}."
        )
    active = _active_gpu_indices(candidate)
    if active:
        observations.append(
            f"Candidate telemetry observed non-zero utilization on GPU indices {active}."
        )
    nvlink = _nested(candidate, "topology", "parsed_matrix", "nvlink_observed")
    if nvlink is False:
        paths = [
            link.get("path")
            for link in _nested(candidate, "topology", "parsed_matrix", "links") or []
        ]
        observations.append(
            "Candidate nvidia-smi topology output contained no NVLink path token; "
            f"observed GPU-to-GPU path(s): {paths or ['unknown']}."
        )

    limitations = [
        "Percentage deltas are observations for these runs, not causal attribution.",
        (
            "This comparison does not isolate collective communication time from "
            "compute, scheduling, graph mode, memory, or other interacting effects."
        ),
    ]
    if not controlled_tp:
        limitations.append(
            "This is not a controlled TP-only comparison because workload or non-TP "
            "engine fields differ."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "comparison_complete",
        "measurement_mode": MEASUREMENT_MODE,
        "baseline": {
            "label": baseline_label,
            "tensor_parallel_size": baseline_tp,
        },
        "candidate": {
            "label": candidate_label,
            "tensor_parallel_size": candidate_tp,
        },
        "comparability": {
            "same_workload": workload_equal,
            "differing_non_tp_engine_fields": differing_engine_fields,
            "controlled_tp_only": controlled_tp,
        },
        "deltas": {
            "output_throughput_percent": throughput_delta,
            "mean_batch_latency_percent": latency_delta,
            "mean_peak_memory_per_device_mib": memory_delta,
            "model_load_time_percent": load_delta,
        },
        "absolute": {
            "baseline_output_tokens_per_second": baseline_aggregate[
                "aggregate_output_tokens_per_second"
            ],
            "candidate_output_tokens_per_second": candidate_aggregate[
                "aggregate_output_tokens_per_second"
            ],
            "baseline_mean_peak_memory_per_device_mib": baseline_memory,
            "candidate_mean_peak_memory_per_device_mib": candidate_memory,
        },
        "observations": observations,
        "limitations": limitations,
    }
