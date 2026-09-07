"""Machine-readable and Markdown output generation for measured M3 evidence."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .errors import ResearchEvidenceError


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format(value: object, digits: int = 3) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_residual_csv(path: Path, fit: Mapping[str, Any]) -> None:
    """Write fitted-versus-observed residual data."""

    rows = fit.get("residuals")
    if not isinstance(rows, list) or not rows:
        raise ResearchEvidenceError("fit has no residual rows")
    fields = (
        "payload_bytes",
        "ring_wire_bytes_per_rank",
        "observed_mean_us",
        "fitted_mean_us",
        "residual_us",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_residual_svg(path: Path, fit: Mapping[str, Any]) -> None:
    """Write a dependency-free residual plot with a non-truncated y axis."""

    rows = fit.get("residuals")
    if not isinstance(rows, list) or not rows:
        raise ResearchEvidenceError("fit has no residual rows")
    width, height, margin = 900, 500, 70
    payloads = [float(row["payload_bytes"]) for row in rows]
    residuals = [float(row["residual_us"]) for row in rows]
    x_min, x_max = min(payloads), max(payloads)
    bound = max(max(abs(value) for value in residuals), 1e-12) * 1.1

    def x_position(payload: float) -> float:
        import math

        return margin + (math.log2(payload) - math.log2(x_min)) / (
            math.log2(x_max) - math.log2(x_min)
        ) * (width - 2 * margin)

    def y_position(residual: float) -> float:
        return height / 2 - residual / bound * (height / 2 - margin)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="450" y="28" text-anchor="middle" font-size="18">'
            f"M3 all-reduce fit residuals (n={len(rows)} payload-level means)</text>"
        ),
        f'<line x1="{margin}" y1="{height/2}" x2="{width-margin}" y2="{height/2}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="black"/>',
        f'<text x="450" y="{height-15}" text-anchor="middle">Payload bytes (log₂ scale)</text>',
        f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle">Residual (µs)</text>',
    ]
    elements.extend(
        f'<circle cx="{x_position(payload):.2f}" cy="{y_position(residual):.2f}" r="5" fill="#1f77b4"/>'
        for payload, residual in zip(payloads, residuals, strict=True)
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def write_comparison_csv(path: Path, comparison: Mapping[str, Any]) -> None:
    """Write the M1/M2/M3 consistency table."""

    rows = comparison.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ResearchEvidenceError("comparison has no rows")
    fields = tuple(rows[0].keys())
    if any(tuple(row.keys()) != fields for row in rows):
        raise ResearchEvidenceError("comparison rows have inconsistent fields")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(
    summaries: Sequence[Mapping[str, Any]],
    fit: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    """Render a concise M3 report without upgrading evidence classifications."""

    lines = [
        "# Milestone 3 — Measured NCCL/PHB Communication Characterization",
        "",
        "Status: **MEASURED_ON_KAGGLE_PENDING_REVIEW**",
        "",
        (
            "This report is valid only when generated from the checksummed raw "
            "two-rank NCCL observations in the same evidence directory. It does "
            "not claim that PHB/NCCL is the sole cause of M1/M2 application behavior."
        ),
        "",
        "## Measured all-reduce summary",
        "",
        "| Payload (bytes) | Samples | Repetitions | Mean (µs) | Median (µs) | p95 (µs) | p99 (µs) | Effective GB/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        latency = row["latency_us"]
        lines.append(
            "| {payload} | {count} | {repetitions} | {mean} | {median} | {p95} | {p99} | {bandwidth} |".format(
                payload=row["payload_bytes"],
                count=latency["count"],
                repetitions=latency["independent_count"],
                mean=_format(latency["mean"]),
                median=_format(latency["median"]),
                p95=_format(latency["p95"]),
                p99=_format(latency["p99"]),
                bandwidth=_format(row["effective_payload_bandwidth_gb_s"]),
            )
        )
    lines.extend(
        [
            "",
            "## Measured communication fit",
            "",
            f"Equation: `{fit['equation']}`",
            "",
            f"Ring volume: `{fit['ring_volume_equation']}`",
            "",
            f"- measured_allreduce_intercept_us: {_format(fit['measured_allreduce_intercept_us'], 6)}",
            f"- beta_effective_gb_s: {_format(fit['beta_effective_gb_s'], 6)}",
            f"- R²: {_format(fit['r_squared'], 6)}",
            "",
            "The intercept is a configuration-specific measured all-reduce proxy. It is not classical transport alpha or universal PCIe latency.",
            "",
            "## M1/M2 consistency comparison",
            "",
            "| Scenario | TP1 output tok/s | TP2 output tok/s | TP2/TP1 | Reciprocal delta (ms/output token) | Payload | Evidence |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in comparison["rows"]:
        lines.append(
            "| {scenario} | {tp1} | {tp2} | {speedup} | {delta} | {payload} | {evidence} |".format(
                scenario=row["scenario_key"],
                tp1=_format(row["tp1_output_tokens_per_second"]),
                tp2=_format(row["tp2_output_tokens_per_second"]),
                speedup=_format(row["tp2_over_tp1_speedup"]),
                delta=_format(
                    row["signed_throughput_reciprocal_delta_ms_per_output_token"]
                ),
                payload=_format(row["collective_payload_bytes"]),
                evidence=row["payload_evidence"],
            )
        )
    lines.extend(["", "## Limitations", ""])
    for limitation in [*fit["limitations"], *comparison["limitations"]]:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def write_m3_outputs(
    output_dir: str | Path,
    *,
    summaries: Sequence[Mapping[str, Any]],
    fit: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> tuple[Path, ...]:
    """Write analysis outputs after all inputs have already validated."""

    target = Path(output_dir)
    if not target.is_dir():
        raise ResearchEvidenceError(f"output directory does not exist: {target}")
    fit_path = target / "M3_FIT.json"
    comparison_path = target / "M3_MODEL_VS_OBSERVED.csv"
    residual_path = target / "M3_FIT_RESIDUALS.csv"
    residual_plot_path = target / "M3_FIT_RESIDUALS.svg"
    report_json_path = target / "M3_REPORT.json"
    report_md_path = target / "M3_REPORT.md"
    report = {
        "schema_version": "kaggle-vllm-m3-report-v1",
        "status": "MEASURED_ON_KAGGLE_PENDING_REVIEW",
        "evidence_classification": {
            "allreduce": "MEASURED",
            "fit": "DERIVED_FROM_MEASURED",
            "m1_m2_throughput": "MEASURED_HISTORICAL",
            "causal_attribution": "UNSUPPORTED",
        },
        "allreduce_summary": list(summaries),
        "communication_fit": dict(fit),
        "model_vs_observed": dict(comparison),
    }
    _write_json(fit_path, fit)
    write_comparison_csv(comparison_path, comparison)
    write_residual_csv(residual_path, fit)
    write_residual_svg(residual_plot_path, fit)
    _write_json(report_json_path, report)
    report_md_path.write_text(
        render_markdown(summaries, fit, comparison), encoding="utf-8"
    )
    return (
        fit_path,
        comparison_path,
        residual_path,
        residual_plot_path,
        report_json_path,
        report_md_path,
    )
