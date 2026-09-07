"""Conservative M1/M2/M3 comparison with explicit evidence classifications."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import ResearchEvidenceError
from .measured_comm import predict_collective_us
from .statistics import finite_number

COMPARISON_SCHEMA = "kaggle-vllm-m3-model-vs-observed-v1"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchEvidenceError(f"cannot parse required evidence: {path}") from error
    if not isinstance(value, dict):
        raise ResearchEvidenceError(f"required evidence is not an object: {path}")
    return value


def _positive_throughput(value: object, *, field: str) -> float:
    throughput = finite_number(value, field=field)
    if throughput <= 0:
        raise ResearchEvidenceError(f"{field} must be positive")
    return throughput


def _payload_fields(
    key: str,
    fit: Mapping[str, Any] | None,
    scenarios: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scenario = scenarios.get(key)
    if scenario is None:
        return {
            "payload_status": "unknown",
            "instantaneous_decode_batch": "unobserved",
            "instantaneous_decode_batch_source": None,
            "collective_payload_bytes": None,
            "payload_derivation": None,
            "m3_predicted_collective_us": None,
            "payload_evidence": "UNSUPPORTED",
        }
    batch_source = str(scenario.get("instantaneous_decode_batch_source", "")).strip()
    if not batch_source:
        raise ResearchEvidenceError(
            f"payload scenario {key!r} requires instantaneous_decode_batch_source"
        )
    batch = scenario.get("instantaneous_decode_batch")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ResearchEvidenceError(
            f"payload scenario {key!r} has invalid instantaneous_decode_batch"
        )
    payload = scenario.get("payload_bytes")
    if isinstance(payload, bool) or not isinstance(payload, int) or payload < 1:
        raise ResearchEvidenceError(f"payload scenario {key!r} has invalid payload_bytes")
    if fit is None:
        predicted = None
        classification = "UNSUPPORTED_WITHOUT_M3_FIT"
    else:
        predicted = predict_collective_us(fit, payload)
        classification = "DERIVED_FROM_MEASURED_M3_FIT"
    return {
        "payload_status": "observed_scheduler_batch_and_derived_payload",
        "instantaneous_decode_batch": batch,
        "instantaneous_decode_batch_source": batch_source,
        "collective_payload_bytes": payload,
        "payload_derivation": str(scenario.get("payload_derivation", "")).strip(),
        "m3_predicted_collective_us": predicted,
        "payload_evidence": classification,
    }


def _comparison_row(
    *,
    key: str,
    milestone: str,
    model: str,
    workload: str,
    tp1: float,
    tp2: float,
    fit: Mapping[str, Any] | None,
    scenarios: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    signed_ms = 1000 * (1 / tp2 - 1 / tp1)
    row = {
        "scenario_key": key,
        "milestone": milestone,
        "model": model,
        "workload": workload,
        "tp1_output_tokens_per_second": tp1,
        "tp2_output_tokens_per_second": tp2,
        "tp2_minus_tp1_output_tokens_per_second": tp2 - tp1,
        "tp2_over_tp1_speedup": tp2 / tp1,
        "signed_throughput_reciprocal_delta_ms_per_output_token": signed_ms,
        "signed_delta_definition": "1000*(1/TP2_output_tok_s - 1/TP1_output_tok_s)",
        "signed_service_time_difference_us": None,
        "signed_service_time_difference_status": (
            "UNSUPPORTED: aggregate throughput does not isolate service time"
        ),
        "observed_regime": (
            "TP2_SLOWER"
            if signed_ms > 0
            else "TP2_FASTER"
            if signed_ms < 0
            else "TP_EQUIVALENT"
        ),
        "throughput_evidence": "MEASURED",
        "signed_delta_evidence": "DERIVED",
        "unexplained_component_us": None,
        "unexplained_component_status": (
            "UNSUPPORTED: application throughput reciprocal is not isolated per-token service time"
        ),
    }
    row.update(_payload_fields(key, fit, scenarios))
    return row


def _load_m1(m1_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("graph", "eager"):
        path = m1_dir / f"comparison-opt125m-{mode}-tp1-vs-tp2.json"
        payload = _load_json(path)
        if payload.get("status") != "comparison_complete":
            raise ResearchEvidenceError(f"M1 comparison is incomplete: {path}")
        absolute = payload.get("absolute")
        if not isinstance(absolute, dict):
            raise ResearchEvidenceError(f"M1 comparison has no absolute metrics: {path}")
        rows.append(
            {
                "key": f"m1-opt125m-{mode}",
                "milestone": "M1",
                "model": "facebook/opt-125m",
                "workload": f"offline_generate_{mode}",
                "tp1": _positive_throughput(
                    absolute.get("baseline_output_tokens_per_second"),
                    field="M1 TP1 throughput",
                ),
                "tp2": _positive_throughput(
                    absolute.get("candidate_output_tokens_per_second"),
                    field="M1 TP2 throughput",
                ),
            }
        )
    return rows


def _load_m2(m2_dir: Path) -> list[dict[str, Any]]:
    summary = _load_json(m2_dir / "summary.json")
    if (
        summary.get("schema_version") != "kaggle-vllm-serving-summary-v1"
        or summary.get("status") != "executed"
    ):
        raise ResearchEvidenceError("M2 summary is incomplete or has an unknown schema")
    matrix = summary.get("matrix")
    if not isinstance(matrix, list):
        raise ResearchEvidenceError("M2 matrix is absent")
    indexed: dict[tuple[int, int], float] = {}
    for raw in matrix:
        if not isinstance(raw, dict):
            raise ResearchEvidenceError("M2 matrix contains a non-object row")
        tp = raw.get("tensor_parallel_size")
        concurrency = raw.get("concurrency")
        if tp not in {1, 2} or isinstance(concurrency, bool) or not isinstance(concurrency, int):
            raise ResearchEvidenceError("M2 matrix has invalid TP/concurrency identity")
        key = (tp, concurrency)
        if key in indexed:
            raise ResearchEvidenceError(f"M2 matrix duplicates cell {key}")
        indexed[key] = _positive_throughput(
            raw.get("output_throughput_tokens_per_second"),
            field=f"M2 throughput {key}",
        )
    expected_concurrency = (1, 4, 8, 16, 32, 64)
    expected = {(tp, concurrency) for tp in (1, 2) for concurrency in expected_concurrency}
    if set(indexed) != expected:
        raise ResearchEvidenceError("M2 matrix is incomplete or contains unexpected cells")
    return [
        {
            "key": f"m2-qwen-c{concurrency:02d}",
            "milestone": "M2",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "workload": f"online_serving_concurrency_{concurrency}",
            "tp1": indexed[(1, concurrency)],
            "tp2": indexed[(2, concurrency)],
        }
        for concurrency in expected_concurrency
    ]


def load_payload_scenarios(path: str | Path | None) -> dict[str, Mapping[str, Any]]:
    """Load optional explicit payload scenarios; no built-in architecture fallback."""

    if path is None:
        return {}
    payload = _load_json(Path(path))
    if payload.get("schema_version") != "kaggle-vllm-m3-payload-scenarios-v1":
        raise ResearchEvidenceError("unsupported payload-scenario schema")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, dict) or not all(
        isinstance(key, str) and isinstance(value, dict)
        for key, value in scenarios.items()
    ):
        raise ResearchEvidenceError("payload scenarios must be an object mapping")
    return scenarios


def build_model_vs_observed(
    m1_dir: str | Path,
    m2_dir: str | Path,
    *,
    fit: Mapping[str, Any] | None,
    payload_scenarios: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed comparison without equating concurrency and batch size."""

    scenarios = payload_scenarios or {}
    source_rows = [*_load_m1(Path(m1_dir)), *_load_m2(Path(m2_dir))]
    rows = [
        _comparison_row(
            key=row["key"],
            milestone=row["milestone"],
            model=row["model"],
            workload=row["workload"],
            tp1=row["tp1"],
            tp2=row["tp2"],
            fit=fit,
            scenarios=scenarios,
        )
        for row in source_rows
    ]
    if not all(math.isfinite(row["tp2_over_tp1_speedup"]) for row in rows):
        raise ResearchEvidenceError("comparison produced a non-finite value")
    return {
        "schema_version": COMPARISON_SCHEMA,
        "status": "comparison_generated",
        "instantaneous_decode_batch": (
            "observed_for_explicit_scenarios_only" if scenarios else "unobserved"
        ),
        "rows": rows,
        "limitations": [
            "Request concurrency is not instantaneous scheduler/decode batch size.",
            "Throughput-reciprocal deltas mix compute, scheduling, memory, communication, and serving effects.",
            "A fitted all-reduce value is consistency evidence, not proof that PHB/NCCL solely caused an application delta.",
            "Unexplained component is intentionally null because the application evidence does not isolate per-token service time.",
        ],
    }
