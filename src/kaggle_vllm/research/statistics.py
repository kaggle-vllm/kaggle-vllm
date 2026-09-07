"""Dependency-free statistical summaries used by research evidence tooling."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from .errors import ResearchEvidenceError


def finite_number(value: object, *, field: str) -> float:
    """Return a finite non-boolean number (including a CSV numeric string)."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ResearchEvidenceError(f"{field} must be a finite number")
    try:
        normalized = float(value)
    except ValueError as error:
        raise ResearchEvidenceError(f"{field} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ResearchEvidenceError(f"{field} must be finite")
    return normalized


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Calculate a nearest-rank percentile for a non-empty finite sample."""

    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    normalized = sorted(finite_number(value, field="measurement") for value in values)
    if not normalized:
        raise ResearchEvidenceError("cannot summarize an empty measurement sample")
    rank = max(0, math.ceil(percentile / 100 * len(normalized)) - 1)
    return normalized[rank]


def t_critical_975(degrees_of_freedom: int) -> float:
    """Return the two-sided 95% Student-t critical value."""

    if degrees_of_freedom < 1:
        raise ResearchEvidenceError("degrees of freedom must be positive")
    table = (
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    if degrees_of_freedom <= len(table):
        return table[degrees_of_freedom - 1]
    if degrees_of_freedom <= 40:
        return 2.021
    if degrees_of_freedom <= 60:
        return 2.000
    if degrees_of_freedom <= 120:
        return 1.980
    return 1.960


def distribution(
    values: Sequence[float], *, independent_values: Sequence[float] | None = None
) -> dict[str, Any]:
    """Summarize measurements and a 95% CI over independent units."""

    normalized = [finite_number(value, field="measurement") for value in values]
    if not normalized:
        raise ResearchEvidenceError("cannot summarize an empty measurement sample")
    independent = [
        finite_number(value, field="independent measurement")
        for value in (independent_values if independent_values is not None else values)
    ]
    if not independent:
        raise ResearchEvidenceError("independent measurement sample is empty")
    mean = statistics.fmean(normalized)
    independent_mean = statistics.fmean(independent)
    independent_std = statistics.stdev(independent) if len(independent) > 1 else 0.0
    if len(independent) > 1:
        half_width = (
            t_critical_975(len(independent) - 1)
            * independent_std
            / math.sqrt(len(independent))
        )
        ci = [independent_mean - half_width, independent_mean + half_width]
    else:
        ci = [None, None]
    return {
        "count": len(normalized),
        "independent_count": len(independent),
        "mean": mean,
        "median": statistics.median(normalized),
        "std": statistics.stdev(normalized) if len(normalized) > 1 else 0.0,
        "p50": nearest_rank(normalized, 50),
        "p95": nearest_rank(normalized, 95),
        "p99": nearest_rank(normalized, 99),
        "minimum": min(normalized),
        "maximum": max(normalized),
        "mean_95_ci": ci,
        "ci_unit": "independent_repetition_means",
    }
