"""Validation, summaries, and linear fits for measured NCCL all-reduce data."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import CommunicationFitError, ResearchEvidenceError
from .statistics import distribution, finite_number, t_critical_975

RAW_SCHEMA = "kaggle-vllm-m3-allreduce-raw-v1"
FIT_SCHEMA = "kaggle-vllm-m3-communication-fit-v1"
REQUIRED_PAYLOAD_BYTES = (
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    33554432,
    67108864,
)
CSV_FIELDS = (
    "schema_version",
    "benchmark_id",
    "payload_bytes",
    "dtype",
    "world_size",
    "repetition",
    "iteration",
    "rank0_latency_us",
    "rank1_latency_us",
    "critical_latency_us",
    "warmup_iterations",
    "timed_iterations",
    "captured_at_utc",
)


@dataclass(frozen=True)
class AllReduceObservation:
    """One timed two-rank collective observation."""

    schema_version: str
    benchmark_id: str
    payload_bytes: int
    dtype: str
    world_size: int
    repetition: int
    iteration: int
    rank0_latency_us: float
    rank1_latency_us: float
    critical_latency_us: float
    warmup_iterations: int
    timed_iterations: int
    captured_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_integer(value: object, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise ResearchEvidenceError(f"{field} must be an integer")
    try:
        normalized = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ResearchEvidenceError(f"{field} must be an integer") from error
    if str(value).strip() not in {str(normalized), f"{normalized}.0"}:
        raise ResearchEvidenceError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if normalized < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ResearchEvidenceError(f"{field} must be {qualifier}")
    return normalized


def _parse_observation(row: Mapping[str, object]) -> AllReduceObservation:
    missing = [field for field in CSV_FIELDS if field not in row]
    if missing:
        raise ResearchEvidenceError(f"all-reduce row missing fields: {missing}")
    if str(row["schema_version"]) != RAW_SCHEMA:
        raise ResearchEvidenceError("unsupported all-reduce raw schema")
    benchmark_id = str(row["benchmark_id"]).strip()
    dtype = str(row["dtype"]).strip()
    captured_at = str(row["captured_at_utc"]).strip()
    if not benchmark_id or not dtype or not captured_at:
        raise ResearchEvidenceError(
            "benchmark_id, dtype, and captured_at_utc must be non-empty"
        )
    world_size = _positive_integer(row["world_size"], field="world_size")
    if world_size != 2:
        raise ResearchEvidenceError("M3 primary schema requires world_size=2")
    rank0 = finite_number(row["rank0_latency_us"], field="rank0_latency_us")
    rank1 = finite_number(row["rank1_latency_us"], field="rank1_latency_us")
    critical = finite_number(
        row["critical_latency_us"], field="critical_latency_us"
    )
    if min(rank0, rank1, critical) <= 0:
        raise ResearchEvidenceError("collective latencies must be positive")
    if not math.isclose(critical, max(rank0, rank1), rel_tol=1e-9, abs_tol=1e-6):
        raise ResearchEvidenceError(
            "critical_latency_us must equal the maximum rank latency"
        )
    return AllReduceObservation(
        schema_version=RAW_SCHEMA,
        benchmark_id=benchmark_id,
        payload_bytes=_positive_integer(row["payload_bytes"], field="payload_bytes"),
        dtype=dtype,
        world_size=world_size,
        repetition=_positive_integer(
            row["repetition"], field="repetition", allow_zero=True
        ),
        iteration=_positive_integer(row["iteration"], field="iteration", allow_zero=True),
        rank0_latency_us=rank0,
        rank1_latency_us=rank1,
        critical_latency_us=critical,
        warmup_iterations=_positive_integer(
            row["warmup_iterations"], field="warmup_iterations", allow_zero=True
        ),
        timed_iterations=_positive_integer(
            row["timed_iterations"], field="timed_iterations"
        ),
        captured_at_utc=captured_at,
    )


def validate_observations(
    observations: Sequence[AllReduceObservation],
    *,
    required_payloads: Iterable[int] = REQUIRED_PAYLOAD_BYTES,
) -> tuple[AllReduceObservation, ...]:
    """Validate identity, payload coverage, and the repetition/iteration grid."""

    if not observations:
        raise ResearchEvidenceError("all-reduce evidence is empty")
    normalized = tuple(_parse_observation(row.to_dict()) for row in observations)
    identities = {
        (row.schema_version, row.benchmark_id, row.dtype, row.world_size)
        for row in normalized
    }
    if len(identities) != 1:
        raise ResearchEvidenceError("all-reduce rows mix benchmark identities")
    required = {_positive_integer(value, field="required payload") for value in required_payloads}
    present = {row.payload_bytes for row in normalized}
    absent = sorted(required - present)
    if absent:
        raise ResearchEvidenceError(f"required payload sizes are absent: {absent}")

    grouped: dict[tuple[int, int], list[AllReduceObservation]] = defaultdict(list)
    for row in normalized:
        grouped[(row.payload_bytes, row.repetition)].append(row)
    repetition_sets: dict[int, set[int]] = defaultdict(set)
    for (payload, repetition), rows in grouped.items():
        repetition_sets[payload].add(repetition)
        expected_count = rows[0].timed_iterations
        if any(row.timed_iterations != expected_count for row in rows):
            raise ResearchEvidenceError("timed_iterations changed within a repetition")
        iterations = sorted(row.iteration for row in rows)
        if iterations != list(range(expected_count)):
            raise ResearchEvidenceError(
                f"incomplete or duplicate iteration grid for payload={payload}, "
                f"repetition={repetition}"
            )
    expected_repetitions = next(iter(repetition_sets.values()))
    if len(expected_repetitions) < 2:
        raise ResearchEvidenceError("at least two independent repetitions are required")
    if expected_repetitions != set(range(len(expected_repetitions))):
        raise ResearchEvidenceError("repetition indices must be contiguous from zero")
    for payload, repetitions in repetition_sets.items():
        if repetitions != expected_repetitions:
            raise ResearchEvidenceError(
                f"incomplete repetition grid for payload={payload}"
            )
    return normalized


def load_allreduce_csv(
    path: str | Path,
    *,
    required_payloads: Iterable[int] = REQUIRED_PAYLOAD_BYTES,
) -> tuple[AllReduceObservation, ...]:
    """Load a strict all-reduce CSV without accepting malformed rows."""

    source = Path(path)
    try:
        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ResearchEvidenceError(
                    f"all-reduce CSV columns must be exactly {list(CSV_FIELDS)}"
                )
            rows = tuple(_parse_observation(row) for row in reader)
    except OSError as error:
        raise ResearchEvidenceError(f"cannot read all-reduce CSV: {source}") from error
    return validate_observations(rows, required_payloads=required_payloads)


def load_allreduce_json(
    path: str | Path,
    *,
    required_payloads: Iterable[int] = REQUIRED_PAYLOAD_BYTES,
) -> tuple[AllReduceObservation, ...]:
    """Load the redundant JSON raw ledger and validate its schema."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchEvidenceError(f"cannot parse all-reduce JSON: {source}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != RAW_SCHEMA:
        raise ResearchEvidenceError("unsupported all-reduce JSON schema")
    records = payload.get("observations")
    if not isinstance(records, list):
        raise ResearchEvidenceError("all-reduce JSON observations must be a list")
    rows = tuple(
        _parse_observation(row)
        for row in records
        if isinstance(row, dict)
    )
    if len(rows) != len(records):
        raise ResearchEvidenceError("all-reduce JSON contains a non-object observation")
    return validate_observations(rows, required_payloads=required_payloads)


def assert_equivalent_ledgers(
    csv_rows: Sequence[AllReduceObservation],
    json_rows: Sequence[AllReduceObservation],
) -> None:
    """Require byte-for-byte field equivalence between redundant raw ledgers."""

    if tuple(csv_rows) != tuple(json_rows):
        raise ResearchEvidenceError("raw CSV and JSON all-reduce ledgers disagree")


def summarize_allreduce(
    observations: Sequence[AllReduceObservation],
) -> list[dict[str, Any]]:
    """Summarize critical-path latency per payload with repetition-level CIs."""

    rows = validate_observations(
        observations,
        required_payloads=sorted({row.payload_bytes for row in observations}),
    )
    grouped: dict[int, list[AllReduceObservation]] = defaultdict(list)
    for row in rows:
        grouped[row.payload_bytes].append(row)
    summaries: list[dict[str, Any]] = []
    for payload_bytes in sorted(grouped):
        selected = grouped[payload_bytes]
        by_repetition: dict[int, list[float]] = defaultdict(list)
        for row in selected:
            by_repetition[row.repetition].append(row.critical_latency_us)
        repetition_means = [
            sum(values) / len(values)
            for _, values in sorted(by_repetition.items())
        ]
        latency = distribution(
            [row.critical_latency_us for row in selected],
            independent_values=repetition_means,
        )
        mean_us = float(latency["mean"])
        bandwidth_values = [
            payload_bytes / row.critical_latency_us / 1000 for row in selected
        ]
        bandwidth_by_repetition = [
            sum(payload_bytes / value / 1000 for value in values) / len(values)
            for _, values in sorted(by_repetition.items())
        ]
        summaries.append(
            {
                "payload_bytes": payload_bytes,
                "world_size": selected[0].world_size,
                "dtype": selected[0].dtype,
                "latency_us": latency,
                "repetition_mean_latency_us": repetition_means,
                "effective_payload_bandwidth_gb_s": payload_bytes / mean_us / 1000,
                "effective_bandwidth_gb_s": distribution(
                    bandwidth_values,
                    independent_values=bandwidth_by_repetition,
                ),
                "bandwidth_definition": (
                    "payload_bytes / critical_path_latency; decimal GB/s"
                ),
            }
        )
    return summaries


def fit_communication_model(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fit T(S)=intercept+wire_bytes/beta to measured payload-level means."""

    if len(summaries) < 3:
        raise CommunicationFitError("at least three payload summaries are required")
    points: list[tuple[float, float, int]] = []
    world_sizes: set[int] = set()
    for index, summary in enumerate(summaries):
        payload = finite_number(summary.get("payload_bytes"), field="payload_bytes")
        if payload <= 0:
            raise CommunicationFitError("payload_bytes must be positive")
        world_size = _positive_integer(summary.get("world_size"), field="world_size")
        world_sizes.add(world_size)
        latency = summary.get("latency_us")
        if not isinstance(latency, Mapping):
            raise CommunicationFitError(f"summary {index} has no latency mapping")
        mean_us = finite_number(latency.get("mean"), field="latency mean")
        if mean_us <= 0:
            raise CommunicationFitError("latency means must be positive")
        ring_factor = 2 * (world_size - 1) / world_size
        points.append((payload * ring_factor, mean_us, int(payload)))
    if world_sizes != {2}:
        raise CommunicationFitError("M3 primary fit requires a two-rank dataset")

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    sxx = sum((value - x_mean) ** 2 for value in x_values)
    if sxx <= 0:
        raise CommunicationFitError("payload sizes must vary")
    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True)
    ) / sxx
    intercept = y_mean - slope * x_mean
    if slope <= 0:
        raise CommunicationFitError(
            "measured fit has non-positive slope; effective bandwidth is undefined"
        )
    predicted = [intercept + slope * value for value in x_values]
    residuals = [
        observed - fitted
        for observed, fitted in zip(y_values, predicted, strict=True)
    ]
    sse = sum(value**2 for value in residuals)
    sst = sum((value - y_mean) ** 2 for value in y_values)
    r_squared = 1 - sse / sst if sst > 0 else None
    degrees_of_freedom = len(points) - 2
    residual_variance = sse / degrees_of_freedom
    critical = t_critical_975(degrees_of_freedom)
    slope_se = math.sqrt(residual_variance / sxx)
    intercept_se = math.sqrt(
        residual_variance * (1 / len(points) + x_mean**2 / sxx)
    )
    slope_ci = [slope - critical * slope_se, slope + critical * slope_se]
    intercept_ci = [
        intercept - critical * intercept_se,
        intercept + critical * intercept_se,
    ]

    def slope_to_bandwidth(value: float) -> float | None:
        return 1 / (value * 1000) if value > 0 else None

    beta = slope_to_bandwidth(slope)
    beta_ci = [slope_to_bandwidth(slope_ci[1]), slope_to_bandwidth(slope_ci[0])]
    residual_rows = [
        {
            "payload_bytes": payload,
            "ring_wire_bytes_per_rank": wire_bytes,
            "observed_mean_us": observed,
            "fitted_mean_us": fitted,
            "residual_us": residual,
        }
        for (wire_bytes, observed, payload), fitted, residual in zip(
            points, predicted, residuals, strict=True
        )
    ]
    return {
        "schema_version": FIT_SCHEMA,
        "evidence_classification": "MEASURED_FIT",
        "equation": "T_us(S)=measured_allreduce_intercept_us + wire_bytes_per_rank / beta_effective",
        "ring_volume_equation": "wire_bytes_per_rank = 2*(P-1)/P*S; for P=2, wire_bytes_per_rank=S",
        "world_size": 2,
        "fit_observation": "payload-level critical-path latency means",
        "point_count": len(points),
        "degrees_of_freedom": degrees_of_freedom,
        "measured_allreduce_intercept_us": intercept,
        "measured_allreduce_intercept_95_ci_us": intercept_ci,
        "slope_us_per_byte": slope,
        "slope_95_ci_us_per_byte": slope_ci,
        "beta_effective_gb_s": beta,
        "beta_effective_95_ci_gb_s": beta_ci,
        "r_squared": r_squared,
        "residual_sum_squares_us2": sse,
        "residuals": residual_rows,
        "limitations": [
            "The intercept is specific to this NCCL/runtime/process/topology/methodology configuration; it is not universal PCIe latency.",
            "A single linear model may not represent NCCL algorithm/protocol transitions across payload regimes.",
            "The fit uses payload-level means; uncertainty in raw collectives and independent repetitions is reported separately.",
            "The fit does not isolate PCIe, PHB, kernel launch, synchronization, or NCCL software contributions.",
        ],
    }


def predict_collective_us(fit: Mapping[str, Any], payload_bytes: int) -> float:
    """Evaluate a measured fit for an explicitly supplied positive payload."""

    payload = _positive_integer(payload_bytes, field="payload_bytes")
    intercept = finite_number(
        fit.get("measured_allreduce_intercept_us"),
        field="measured_allreduce_intercept_us",
    )
    slope = finite_number(fit.get("slope_us_per_byte"), field="slope_us_per_byte")
    if slope <= 0:
        raise CommunicationFitError("slope_us_per_byte must be positive")
    return intercept + payload * slope
