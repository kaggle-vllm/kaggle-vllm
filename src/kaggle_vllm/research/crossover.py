"""Fail-closed CPU analysis for the planned M4 TP crossover matrix."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .errors import ResearchEvidenceError
from .statistics import distribution, finite_number

M4_RAW_SCHEMA = "kaggle-vllm-m4-serving-raw-v1"
PRINCIPAL_CONCURRENCY = (1, 4, 8, 16, 32, 64)
WORKLOAD_TOKENS = {
    "short": (128, 64),
    "balanced": (512, 256),
    "prefill_heavy": (2048, 128),
}
REQUIRED_FIELDS = {
    "schema_version",
    "model_id",
    "model_revision",
    "workload",
    "input_tokens",
    "output_tokens_requested",
    "tensor_parallel_size",
    "concurrency",
    "repetition",
    "request_throughput_per_second",
    "input_tokens_per_second",
    "output_tokens_per_second",
    "total_tokens_per_second",
    "ttft_ms",
    "tpot_ms",
    "itl_ms",
    "e2e_latency_ms",
    "request_failures",
    "oom",
    "warmup_requests",
    "measured_requests",
    "preemptions",
    "kv_cache_occupancy_percent",
    "gpu_utilization_percent",
    "maximum_vram_mib",
    "maximum_system_ram_bytes",
    "mean_power_w",
    "maximum_temperature_c",
    "benchmark_tool",
    "metric_definitions",
    "prompt_manifest_sha256",
}


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchEvidenceError(f"cannot parse M4 evidence: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != M4_RAW_SCHEMA:
        raise ResearchEvidenceError("unsupported M4 raw schema")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ResearchEvidenceError("M4 raw evidence must contain rows")
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict) or set(raw) != REQUIRED_FIELDS:
            raise ResearchEvidenceError(f"M4 row {index} has missing or extra fields")
        workload = raw["workload"]
        if workload not in WORKLOAD_TOKENS:
            raise ResearchEvidenceError(f"M4 row {index} has unknown workload")
        if (raw["input_tokens"], raw["output_tokens_requested"]) != WORKLOAD_TOKENS[workload]:
            raise ResearchEvidenceError(f"M4 row {index} token counts do not match protocol")
        if raw["tensor_parallel_size"] not in {1, 2}:
            raise ResearchEvidenceError(f"M4 row {index} has invalid TP")
        if isinstance(raw["concurrency"], bool) or not isinstance(raw["concurrency"], int) or raw["concurrency"] < 1:
            raise ResearchEvidenceError(f"M4 row {index} has invalid concurrency")
        if isinstance(raw["repetition"], bool) or not isinstance(raw["repetition"], int) or raw["repetition"] < 0:
            raise ResearchEvidenceError(f"M4 row {index} has invalid repetition")
        if isinstance(raw["request_failures"], bool) or not isinstance(raw["request_failures"], int) or raw["request_failures"] < 0:
            raise ResearchEvidenceError(f"M4 row {index} has invalid failure count")
        if not isinstance(raw["oom"], bool):
            raise ResearchEvidenceError(f"M4 row {index} oom must be boolean")
        for field in ("warmup_requests", "measured_requests"):
            if isinstance(raw[field], bool) or not isinstance(raw[field], int) or raw[field] < 1:
                raise ResearchEvidenceError(f"M4 row {index} has invalid {field}")
        if raw["preemptions"] is not None and (
            isinstance(raw["preemptions"], bool)
            or not isinstance(raw["preemptions"], int)
            or raw["preemptions"] < 0
        ):
            raise ResearchEvidenceError(f"M4 row {index} has invalid preemptions")
        for field in (
            "kv_cache_occupancy_percent", "gpu_utilization_percent",
            "maximum_vram_mib", "maximum_system_ram_bytes", "mean_power_w",
            "maximum_temperature_c",
        ):
            if raw[field] is not None and finite_number(raw[field], field=field) < 0:
                raise ResearchEvidenceError(f"M4 row {index} {field} cannot be negative")
        if raw["maximum_vram_mib"] is not None and raw["maximum_vram_mib"] > 14.5 * 1024:
            raise ResearchEvidenceError(f"M4 row {index} exceeded the GPU memory cap")
        if raw["maximum_system_ram_bytes"] is not None and raw["maximum_system_ram_bytes"] > 28 * 1024**3:
            raise ResearchEvidenceError(f"M4 row {index} exceeded the RAM cap")
        if not isinstance(raw["benchmark_tool"], str) or not raw["benchmark_tool"].strip():
            raise ResearchEvidenceError(f"M4 row {index} lacks benchmark tool identity")
        if not isinstance(raw["metric_definitions"], dict) or not raw["metric_definitions"]:
            raise ResearchEvidenceError(f"M4 row {index} lacks metric definitions")
        if not re.fullmatch(r"[0-9a-f]{64}", str(raw["prompt_manifest_sha256"])):
            raise ResearchEvidenceError(f"M4 row {index} has invalid prompt manifest hash")
        failed = raw["oom"] or raw["request_failures"] > 0
        for field in (
            "request_throughput_per_second", "input_tokens_per_second",
            "output_tokens_per_second", "total_tokens_per_second", "ttft_ms",
            "tpot_ms", "itl_ms", "e2e_latency_ms",
        ):
            if raw[field] is None and failed:
                continue
            value = finite_number(raw[field], field=field)
            if value <= 0:
                raise ResearchEvidenceError(f"M4 row {index} {field} must be positive")
        identity = (
            raw["model_id"], raw["model_revision"], workload,
            raw["tensor_parallel_size"], raw["concurrency"], raw["repetition"],
        )
        if identity in identities:
            raise ResearchEvidenceError(f"M4 duplicate repetition: {identity}")
        identities.add(identity)
        normalized.append(raw)
    return normalized


def analyze_m4(path: str | Path, *, minimum_repetitions: int = 5) -> dict[str, Any]:
    """Summarize matched TP1/TP2 repetitions and classify crossover evidence."""

    rows = _load_rows(path)
    groups: dict[tuple[str, str, str, int], dict[int, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        key = (row["model_id"], row["model_revision"], row["workload"], row["concurrency"])
        groups[key][row["tensor_parallel_size"]][row["repetition"]] = row
    results = []
    for key, by_tp in sorted(groups.items()):
        if set(by_tp) != {1, 2}:
            raise ResearchEvidenceError(f"M4 cell lacks TP1 or TP2: {key}")
        repetitions = set(by_tp[1])
        if repetitions != set(by_tp[2]) or len(repetitions) < minimum_repetitions:
            raise ResearchEvidenceError(f"M4 cell lacks matched independent repetitions: {key}")
        tp1_capacity_failure = all(
            by_tp[1][rep]["oom"] or by_tp[1][rep]["request_failures"] > 0
            for rep in repetitions
        )
        tp2_capacity_pass = all(
            not by_tp[2][rep]["oom"] and by_tp[2][rep]["request_failures"] == 0
            for rep in repetitions
        )
        metrics = (
            "request_throughput_per_second",
            "input_tokens_per_second",
            "output_tokens_per_second",
            "total_tokens_per_second",
            "ttft_ms",
            "tpot_ms",
            "itl_ms",
            "e2e_latency_ms",
            "request_failures",
            "preemptions",
            "kv_cache_occupancy_percent",
            "gpu_utilization_percent",
            "maximum_vram_mib",
            "maximum_system_ram_bytes",
            "mean_power_w",
            "maximum_temperature_c",
        )
        tp_summaries = {}
        for tp in (1, 2):
            tp_summaries[f"tp{tp}"] = {}
            for metric in metrics:
                values = [by_tp[tp][rep][metric] for rep in sorted(repetitions)]
                tp_summaries[f"tp{tp}"][metric] = (
                    distribution(values)
                    if all(value is not None for value in values)
                    else None
                )
        throughput_stats = None
        latency_stats = None
        speedup_stats = None
        comparison_metrics_available = all(
            by_tp[tp][rep][metric] is not None
            for tp in (1, 2)
            for rep in repetitions
            for metric in ("output_tokens_per_second", "e2e_latency_ms")
        )
        if not tp1_capacity_failure and comparison_metrics_available:
            throughput_delta = [
                by_tp[2][rep]["output_tokens_per_second"] - by_tp[1][rep]["output_tokens_per_second"]
                for rep in sorted(repetitions)
            ]
            latency_delta = [
                by_tp[2][rep]["e2e_latency_ms"] - by_tp[1][rep]["e2e_latency_ms"]
                for rep in sorted(repetitions)
            ]
            speedup = [
                by_tp[2][rep]["output_tokens_per_second"] / by_tp[1][rep]["output_tokens_per_second"]
                for rep in sorted(repetitions)
            ]
            throughput_stats = distribution(throughput_delta)
            latency_stats = distribution(latency_delta)
            speedup_stats = distribution(speedup)
        throughput_robust = (
            throughput_stats is not None and throughput_stats["mean_95_ci"][0] > 0
        )
        latency_robust = (
            latency_stats is not None and latency_stats["mean_95_ci"][1] < 0
        )
        classifications = []
        if tp1_capacity_failure and tp2_capacity_pass:
            classifications.append("CAPACITY_CROSSOVER")
        if throughput_robust and not tp1_capacity_failure:
            classifications.append("THROUGHPUT_CROSSOVER")
        if latency_robust and not tp1_capacity_failure:
            classifications.append("LATENCY_CROSSOVER")
        if not classifications:
            classifications.append("NO_CROSSOVER_OBSERVED")
        results.append(
            {
                "model_id": key[0], "model_revision": key[1], "workload": key[2],
                "concurrency": key[3], "repetitions": len(repetitions),
                "tp2_minus_tp1_output_tokens_per_second": throughput_stats,
                "tp2_minus_tp1_e2e_latency_ms": latency_stats,
                "tp2_over_tp1_output_speedup": speedup_stats,
                "tp1": tp_summaries["tp1"],
                "tp2": tp_summaries["tp2"],
                "classifications": classifications,
                "evidence": "MEASURED_INPUT_DERIVED_COMPARISON",
            }
        )
    by_series: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in results:
        by_series[(cell["model_id"], cell["model_revision"], cell["workload"])].append(cell)
    crossover_summary = []
    for key, series in sorted(by_series.items()):
        ordered = sorted(series, key=lambda item: item["concurrency"])
        observed = [item["concurrency"] for item in ordered]

        def sustained(label: str, cells: list[dict[str, Any]] = ordered) -> int | None:
            flags = [label in item["classifications"] for item in cells]
            for index, enabled in enumerate(flags):
                if enabled and all(flags[index:]):
                    return cells[index]["concurrency"]
            return None

        complete = observed == list(PRINCIPAL_CONCURRENCY)
        crossover_summary.append(
            {
                "model_id": key[0],
                "model_revision": key[1],
                "workload": key[2],
                "observed_concurrency": observed,
                "principal_grid_complete": complete,
                "throughput_crossover_concurrency": sustained("THROUGHPUT_CROSSOVER") if complete else None,
                "latency_crossover_concurrency": sustained("LATENCY_CROSSOVER") if complete else None,
                "capacity_crossover_concurrency": sustained("CAPACITY_CROSSOVER") if complete else None,
                "criterion": "first robust favorable point sustained through all higher frozen concurrency points",
            }
        )
    return {
        "schema_version": "kaggle-vllm-m4-analysis-v1",
        "status": "ANALYZED",
        "crossover_criterion": "paired-repetition mean-delta 95% CI excludes zero in the favorable direction",
        "cells": results,
        "crossover_summary": crossover_summary,
    }
