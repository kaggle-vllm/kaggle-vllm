#!/usr/bin/env python3
"""Generate paper figures/tables directly from committed evidence."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def save_figure(figure: Any, destination: Path, stem: str) -> None:
    for suffix in ("svg", "png", "pdf"):
        figure.savefig(destination / f"{stem}.{suffix}", bbox_inches="tight", dpi=240)


def m1_data(root: Path) -> list[dict[str, Any]]:
    rows = []
    for mode in ("graph", "eager"):
        source = load_object(root / f"comparison-opt125m-{mode}-tp1-vs-tp2.json")
        for role, tp in (("baseline", 1), ("candidate", 2)):
            run = load_object(root / f"{source[role]['label']}.json")
            trial = run["measurements"]["aggregate"]["trial_output_tokens_per_second"]
            rows.append(
                {
                    "mode": mode,
                    "tp": tp,
                    "output_tokens_per_second": trial["mean"],
                    "sample_standard_deviation": trial["sample_standard_deviation"],
                    "trial_count": trial["count"],
                }
            )
    return rows


def m2_data(root: Path) -> list[dict[str, Any]]:
    payload = load_object(root / "summary.json")
    if payload.get("status") != "executed" or not isinstance(payload.get("matrix"), list):
        raise ValueError("M2 summary is not executed/complete")
    return payload["matrix"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def svg_plot(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[tuple[str, list[tuple[float, float, float | None]]]],
) -> None:
    """Write a small dependency-free SVG when Matplotlib is unavailable."""

    width, height = 900, 540
    left, right, top, bottom = 90, 30, 65, 80
    points = [(point[0], point[1]) for _, values in series for point in values]
    if not points:
        raise ValueError("cannot plot empty series")
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = 0.0, max(y for _, y in points) * 1.1
    if xmax == xmin:
        xmax = xmin + 1
    colors = ("#1f77b4", "#d62728", "#2ca02c")

    def transform(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        return (
            left + (x - xmin) / (xmax - xmin) * (width - left - right),
            height - bottom - (y - ymin) / (ymax - ymin) * (height - top - bottom),
        )

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="18">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-20}" text-anchor="middle">{html.escape(xlabel)}</text>',
        f'<text transform="translate(20 {height/2}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>',
    ]
    for index, (label, values) in enumerate(series):
        transformed = [transform((point[0], point[1])) for point in values]
        polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in transformed)
        color = colors[index % len(colors)]
        elements.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="3"/>')
        elements.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>' for x, y in transformed)
        for point, (x, _) in zip(values, transformed, strict=True):
            if point[2] is not None:
                upper = transform((point[0], point[1] + point[2]))[1]
                lower = transform((point[0], max(0, point[1] - point[2])))[1]
                elements.append(
                    f'<line x1="{x:.2f}" y1="{upper:.2f}" x2="{x:.2f}" '
                    f'y2="{lower:.2f}" stroke="{color}" stroke-width="2"/>'
                )
        elements.append(f'<text x="{width-180}" y="{55+index*22}" fill="{color}">{html.escape(label)}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def generate_svg_fallback(args: argparse.Namespace) -> dict[str, str]:
    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    first = m1_data(args.m1)
    second = m2_data(args.m2)
    write_csv(args.tables / "m1_tp1_tp2.csv", first)
    write_csv(args.tables / "m2_qwen_crossover.csv", second)
    svg_plot(
        args.figures / "01_m1_low_load.svg",
        title="M1 measured low-load TP behavior (n=5 trials/cell; ±1 SD)",
        xlabel="Graph=0–1; eager=2–3 (TP1 then TP2)",
        ylabel="Output tokens/s",
        series=[("M1", [(float(index), row["output_tokens_per_second"], row["sample_standard_deviation"]) for index, row in enumerate(first)])],
    )
    svg_plot(
        args.figures / "02_m2_qwen_crossover.svg",
        title="M2 measured Qwen concurrency crossover (n=1 run/cell)",
        xlabel="Request concurrency (not scheduler batch size)",
        ylabel="Output tokens/s",
        series=[
            (
                f"TP={tp}",
                [(row["concurrency"], row["output_throughput_tokens_per_second"], None) for row in sorted((item for item in second if item["tensor_parallel_size"] == tp), key=lambda item: item["concurrency"])],
            )
            for tp in (1, 2)
        ],
    )
    status = {
        "m1": "GENERATED_SVG_FROM_MEASURED_EVIDENCE",
        "m2": "GENERATED_SVG_FROM_MEASURED_EVIDENCE",
        "m3": "UNSUPPORTED_NO_M3_EVIDENCE" if args.m3 is None else "MATPLOTLIB_REQUIRED_FOR_M3",
        "m4": "UNSUPPORTED_NO_M4_EVIDENCE" if args.m4 is None else "MATPLOTLIB_REQUIRED_FOR_M4",
        "m5": "UNSUPPORTED_NO_VALID_SIMULATOR_EVIDENCE" if args.m5 is None else "MATPLOTLIB_REQUIRED_FOR_M5",
        "format_note": "Matplotlib unavailable; dependency-free SVG generated. Install research-only Matplotlib for PNG/PDF.",
    }
    (args.figures / "generation-status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def generate(args: argparse.Namespace) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return generate_svg_fallback(args)

    args.figures.mkdir(parents=True, exist_ok=True)
    args.tables.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    first = m1_data(args.m1)
    write_csv(args.tables / "m1_tp1_tp2.csv", first)
    figure, axis = plt.subplots()
    labels = [f"{row['mode']} TP{row['tp']}" for row in first]
    axis.bar(
        labels,
        [row["output_tokens_per_second"] for row in first],
        yerr=[row["sample_standard_deviation"] for row in first],
        capsize=4,
    )
    axis.set_ylabel("Output tokens/s")
    axis.set_ylim(bottom=0)
    axis.set_title("M1 measured low-load TP behavior (n=5 trials/cell; ±1 SD)")
    save_figure(figure, args.figures, "01_m1_low_load")
    plt.close(figure)
    status["m1"] = "GENERATED_FROM_MEASURED_EVIDENCE"

    second = m2_data(args.m2)
    write_csv(args.tables / "m2_qwen_crossover.csv", second)
    figure, axis = plt.subplots()
    for tp in (1, 2):
        rows = sorted((row for row in second if row["tensor_parallel_size"] == tp), key=lambda row: row["concurrency"])
        axis.plot([row["concurrency"] for row in rows], [row["output_throughput_tokens_per_second"] for row in rows], marker="o", label=f"TP={tp}")
    axis.set_xlabel("Request concurrency (not scheduler batch size)")
    axis.set_ylabel("Output tokens/s")
    axis.set_ylim(bottom=0)
    axis.set_title("M2 measured Qwen concurrency crossover (n=1 run/cell)")
    axis.legend()
    save_figure(figure, args.figures, "02_m2_qwen_crossover")
    plt.close(figure)
    status["m2"] = "GENERATED_FROM_MEASURED_EVIDENCE"

    if args.m3 is None:
        status["m3"] = "UNSUPPORTED_NO_M3_EVIDENCE"
    else:
        report = load_object(args.m3 / "M3_REPORT.json")
        summaries = report.get("allreduce_summary")
        fit = load_object(args.m3 / "M3_FIT.json")
        if not isinstance(summaries, list) or not summaries:
            raise ValueError("M3 report has no summaries")
        write_csv(args.tables / "m3_allreduce_summary.csv", summaries)
        payloads = [row["payload_bytes"] for row in summaries]
        latency_stats = [row["latency_us"] for row in summaries]
        bandwidth_stats = [row["effective_bandwidth_gb_s"] for row in summaries]
        repetitions = min(row["independent_count"] for row in latency_stats)
        for stem, stats, ylabel in (
            ("03_m3_latency", latency_stats, "All-reduce critical latency (µs)"),
            ("04_m3_bandwidth", bandwidth_stats, "Effective bandwidth (GB/s)"),
        ):
            figure, axis = plt.subplots()
            means = [row["mean"] for row in stats]
            intervals = [row["mean_95_ci"] for row in stats]
            axis.errorbar(
                payloads,
                means,
                yerr=[
                    [mean - interval[0] for mean, interval in zip(means, intervals, strict=True)],
                    [interval[1] - mean for mean, interval in zip(means, intervals, strict=True)],
                ],
                marker="o",
                capsize=3,
            )
            axis.set_xscale("log", base=2); axis.set_ylim(bottom=0)
            axis.set_xlabel("Payload bytes (log₂ scale)"); axis.set_ylabel(ylabel)
            axis.set_title(f"M3 measured two-rank NCCL all-reduce (n={repetitions} repetitions; 95% CI)")
            save_figure(figure, args.figures, stem); plt.close(figure)
        residuals = fit.get("residuals")
        if not isinstance(residuals, list) or not residuals:
            raise ValueError("M3 fit has no residuals")
        write_csv(args.tables / "m3_fit_residuals.csv", residuals)
        figure, axis = plt.subplots()
        axis.axhline(0, color="black", linewidth=0.8)
        axis.scatter([row["payload_bytes"] for row in residuals], [row["residual_us"] for row in residuals])
        axis.set_xscale("log", base=2); axis.set_xlabel("Payload bytes (log₂ scale)"); axis.set_ylabel("Fit residual (µs)")
        axis.set_title("M3 all-reduce fit residuals")
        save_figure(figure, args.figures, "05_m3_fit_residuals"); plt.close(figure)
        status["m3"] = "GENERATED_FROM_MEASURED_EVIDENCE"

    if args.m4 is None:
        status["m4"] = "UNSUPPORTED_NO_M4_EVIDENCE"
    else:
        analysis = load_object(args.m4)
        cells = analysis.get("cells")
        if not isinstance(cells, list) or not cells:
            raise ValueError("M4 analysis has no cells")
        def mean_or_none(value: Any) -> Any:
            return value.get("mean") if isinstance(value, dict) else None

        flat_cells = [
            {
                "model_id": cell["model_id"],
                "model_revision": cell["model_revision"],
                "workload": cell["workload"],
                "concurrency": cell["concurrency"],
                "repetitions": cell["repetitions"],
                "tp1_output_tokens_per_second": mean_or_none(cell["tp1"]["output_tokens_per_second"]),
                "tp2_output_tokens_per_second": mean_or_none(cell["tp2"]["output_tokens_per_second"]),
                "tp2_over_tp1_speedup": mean_or_none(cell["tp2_over_tp1_output_speedup"]),
                "tp1_ttft_ms": mean_or_none(cell["tp1"]["ttft_ms"]),
                "tp2_ttft_ms": mean_or_none(cell["tp2"]["ttft_ms"]),
                "tp1_tpot_ms": mean_or_none(cell["tp1"]["tpot_ms"]),
                "tp2_tpot_ms": mean_or_none(cell["tp2"]["tpot_ms"]),
                "classifications": ";".join(cell["classifications"]),
            }
            for cell in cells
        ]
        write_csv(args.tables / "m4_crossover_cells.csv", flat_cells)
        models = sorted({cell["model_id"] for cell in cells})
        workloads = sorted({cell["workload"] for cell in cells})
        minimum_repetitions = min(cell["repetitions"] for cell in cells)
        for stem, metric, ylabel in (
            ("06_m4_multimodel_crossover", "output_tokens_per_second", "Output tokens/s"),
            ("08_m4_ttft", "ttft_ms", "TTFT (ms)"),
            ("09_m4_tpot", "tpot_ms", "TPOT (ms/token)"),
        ):
            figure, axis = plt.subplots(figsize=(10, 6))
            for model in models:
                for workload in workloads:
                    selected = sorted(
                        (
                            cell for cell in cells
                            if cell["model_id"] == model
                            and cell["workload"] == workload
                            and cell["tp1"][metric] is not None
                            and cell["tp2"][metric] is not None
                        ),
                        key=lambda cell: cell["concurrency"],
                    )
                    for tp in (1, 2):
                        if not selected:
                            continue
                        means = [cell[f"tp{tp}"][metric]["mean"] for cell in selected]
                        intervals = [cell[f"tp{tp}"][metric]["mean_95_ci"] for cell in selected]
                        axis.errorbar(
                            [cell["concurrency"] for cell in selected], means,
                            yerr=[
                                [mean - interval[0] for mean, interval in zip(means, intervals, strict=True)],
                                [interval[1] - mean for mean, interval in zip(means, intervals, strict=True)],
                            ],
                            marker="o", capsize=2, label=f"{model} / {workload} / TP{tp}",
                        )
            axis.set_xlabel("Request concurrency (not scheduler batch size)")
            axis.set_ylabel(ylabel); axis.set_ylim(bottom=0)
            axis.set_title(f"M4 measured serving results (n≥{minimum_repetitions} independent repetitions; 95% CI)")
            axis.legend(fontsize=7, ncol=2)
            save_figure(figure, args.figures, stem); plt.close(figure)

        columns = [(workload, concurrency) for workload in workloads for concurrency in sorted({cell["concurrency"] for cell in cells})]
        matrix = []
        for model in models:
            indexed = {
                (cell["workload"], cell["concurrency"]): mean_or_none(
                    cell["tp2_over_tp1_output_speedup"]
                )
                for cell in cells
                if cell["model_id"] == model
            }
            matrix.append([
                indexed.get(column) if indexed.get(column) is not None else float("nan")
                for column in columns
            ])
        finite = [value for row in matrix for value in row if math.isfinite(value)]
        span = max(max((abs(value - 1) for value in finite), default=0.0), 0.01)
        figure, axis = plt.subplots(figsize=(max(10, len(columns) * 0.6), max(3, len(models) * 0.7)))
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=1 - span, vmax=1 + span)
        axis.set_xticks(range(len(columns)), [f"{workload}\nc={concurrency}" for workload, concurrency in columns], rotation=45, ha="right")
        axis.set_yticks(range(len(models)), models)
        axis.set_title(f"M4 TP2/TP1 output-throughput speedup (n≥{minimum_repetitions}; 1.0=no change)")
        figure.colorbar(image, ax=axis, label="TP2 / TP1 speedup")
        save_figure(figure, args.figures, "07_m4_speedup_heatmap"); plt.close(figure)
        status["m4"] = "FIGURES_AND_TABLE_GENERATED_FROM_MEASURED_EVIDENCE"
    if args.m5 is None:
        status["m5"] = "UNSUPPORTED_NO_VALID_SIMULATOR_EVIDENCE"
    else:
        simulation = load_object(args.m5)
        if simulation.get("schema_version") != "kaggle-vllm-m5-simulator-comparison-v1":
            raise ValueError("unsupported M5 simulator-comparison schema")
        simulation_rows = simulation.get("rows")
        if not isinstance(simulation_rows, list) or not simulation_rows:
            raise ValueError("M5 simulator comparison has no rows")
        if any(row.get("calibration_role") != "independent_validation" for row in simulation_rows):
            raise ValueError("M5 paper plot requires independent validation rows")
        write_csv(args.tables / "m5_simulator_vs_observed.csv", simulation_rows)
        figure, axis = plt.subplots()
        observed = [row["observed"] for row in simulation_rows]
        predicted = [row["predicted"] for row in simulation_rows]
        lower = min([*observed, *predicted, 0])
        upper = max([*observed, *predicted])
        axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", label="perfect prediction")
        axis.scatter(observed, predicted)
        axis.set_xlabel("Observed value (row-specific unit)")
        axis.set_ylabel("Predicted value (same row-specific unit)")
        axis.set_title(f"M5 simulator validation (n={len(simulation_rows)} independent cells)")
        axis.legend()
        save_figure(figure, args.figures, "10_m5_simulator_vs_observed"); plt.close(figure)
        status["m5"] = "GENERATED_FROM_INDEPENDENT_VALIDATION_EVIDENCE"
    (args.figures / "generation-status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--m2", type=Path, required=True)
    parser.add_argument("--m3", type=Path)
    parser.add_argument("--m4", type=Path)
    parser.add_argument("--m5", type=Path)
    parser.add_argument("--figures", type=Path, default=Path("research/figures"))
    parser.add_argument("--tables", type=Path, default=Path("research/tables"))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(generate(parse_args()), indent=2))
