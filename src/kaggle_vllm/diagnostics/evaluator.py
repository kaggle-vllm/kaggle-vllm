"""Load M1/M2 evidence and attach communication-cost diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kaggle_vllm.diagnostics.alpha_beta_model import (
    AlphaBetaCommModel,
    CellCommDiagnosis,
)


@dataclass(frozen=True)
class CrossoverAnalysis:
    """Measured crossover detection vs analytical prediction status."""

    measured_throughput_crossover_concurrency: int | None
    measured_crossover_tp1_tok_s: float | None
    measured_crossover_tp2_tok_s: float | None
    model_predicted_crossover_concurrency: int | None
    prediction_status: str
    prediction_reason: str


@dataclass(frozen=True)
class MilestoneEvaluation:
    milestone: str
    cells: list[CellCommDiagnosis]
    crossover: CrossoverAnalysis | None
    notes: list[str]


class MilestoneArtifactEvaluator:
    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        enable_hypothetical: bool = False,
    ) -> None:
        self.artifact_path = Path(artifact_dir)
        self.model = AlphaBetaCommModel(enable_hypothetical=enable_hypothetical)

    def parse_m1_evidence(self) -> MilestoneEvaluation:
        cells: list[CellCommDiagnosis] = []
        notes: list[str] = [
            "M1 offline comparisons: treat graph vs eager proxy spread as a stability warning.",
            "M1 text already states communication was not causally isolated.",
        ]

        for comp_file in sorted(self.artifact_path.glob("comparison-*.json")):
            try:
                data = json.loads(comp_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeError):
                continue

            label = comp_file.stem
            abs_data = data.get("absolute", {})
            tp1 = float(abs_data.get("baseline_output_tokens_per_second", 0.0) or 0.0)
            tp2 = float(abs_data.get("candidate_output_tokens_per_second", 0.0) or 0.0)

            is_tp = True
            model_key = "opt-125m" if "opt125m" in label else "qwen2.5-3b"
            if "batching" in label:
                is_tp = False

            # Prefer engine metadata when present on side files — comparisons encode TP in name
            if "opt125m-graph" in label or "opt125m-eager" in label:
                model_key = "opt-125m"

            cells.append(
                self.model.diagnose_cell(
                    label=label,
                    model_key=model_key,
                    tp1_tok_s=tp1,
                    tp2_tok_s=tp2,
                    concurrency=1,
                    comparison_is_tp_controlled=is_tp,
                )
            )

        # Explicit note on proxy spread graph vs eager
        graph_proxy = next(
            (
                c.observed_excess_us_per_collective_proxy
                for c in cells
                if c.label == "comparison-opt125m-graph-tp1-vs-tp2"
                and c.observed_excess_us_per_collective_proxy is not None
            ),
            None,
        )
        eager_proxy = next(
            (
                c.observed_excess_us_per_collective_proxy
                for c in cells
                if c.label == "comparison-opt125m-eager-tp1-vs-tp2"
                and c.observed_excess_us_per_collective_proxy is not None
            ),
            None,
        )
        if graph_proxy is not None and eager_proxy is not None:
            notes.append(
                f"OPT-125M excess-time proxy differs sharply: "
                f"graph~{graph_proxy:.2f} us/collective vs eager~{eager_proxy:.2f} us/collective. "
                f"This spread indicates the proxy is not a stable transport-latency alpha."
            )

        return MilestoneEvaluation(
            milestone="m1",
            cells=cells,
            crossover=None,
            notes=notes,
        )

    def parse_m2_concurrency_matrix(self) -> MilestoneEvaluation:
        notes: list[str] = [
            "M2 online serving: concurrency is NOT proven equal to instantaneous decode batch.",
            "Crossover c=16 is MEASURED detection from the matrix, not an alpha-beta prediction.",
        ]
        summary_path = self.artifact_path / "summary.json"
        cells: list[CellCommDiagnosis] = []

        if not summary_path.exists():
            return MilestoneEvaluation("m2", [], None, notes + ["summary.json missing"])

        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return MilestoneEvaluation(
                "m2", [], None, notes + ["summary.json unreadable"]
            )

        matrix = data.get("matrix")
        if not isinstance(matrix, list):
            return MilestoneEvaluation("m2", [], None, notes + ["matrix missing"])

        by_c: dict[int, dict[int, dict]] = {}
        for row in matrix:
            if not isinstance(row, dict):
                continue
            try:
                c = int(row["concurrency"])
                tp = int(row["tensor_parallel_size"])
            except (KeyError, TypeError, ValueError):
                continue
            by_c.setdefault(c, {})[tp] = row

        for c in sorted(by_c):
            r1, r2 = by_c[c].get(1), by_c[c].get(2)
            if not r1 or not r2:
                continue
            tp1 = float(r1.get("output_throughput_tokens_per_second", 0.0) or 0.0)
            tp2 = float(r2.get("output_throughput_tokens_per_second", 0.0) or 0.0)
            cells.append(
                self.model.diagnose_cell(
                    label=f"qwen-tp-c{c:02d}",
                    model_key="qwen2.5-3b",
                    tp1_tok_s=tp1,
                    tp2_tok_s=tp2,
                    concurrency=c,
                    comparison_is_tp_controlled=True,
                )
            )

        measured_cross = None
        t1 = t2 = None
        for cell in cells:
            if cell.measured_tp2_tok_s > cell.measured_tp1_tok_s:
                measured_cross = cell.concurrency
                t1, t2 = cell.measured_tp1_tok_s, cell.measured_tp2_tok_s
                break

        analysis = (
            data.get("analysis") if isinstance(data.get("analysis"), dict) else {}
        )
        declared = analysis.get("throughput_crossover_concurrency")
        if (
            declared is not None
            and measured_cross is not None
            and int(declared) != measured_cross
        ):
            notes.append(
                f"summary.analysis crossover={declared} differs from recomputed first cross={measured_cross}."
            )

        crossover = CrossoverAnalysis(
            measured_throughput_crossover_concurrency=measured_cross,
            measured_crossover_tp1_tok_s=t1,
            measured_crossover_tp2_tok_s=t2,
            model_predicted_crossover_concurrency=None,
            prediction_status="unsupported",
            prediction_reason=(
                "M1/M2 evidence lacks isolated collective timestamps and true per-step "
                "scheduler batch sizes. An α–β model therefore cannot defensibly *predict* "
                "the concurrency crossover; c=16 is reported as an observed matrix fact only."
            ),
        )

        return MilestoneEvaluation(
            milestone="m2", cells=cells, crossover=crossover, notes=notes
        )


def evaluation_to_jsonable(ev: MilestoneEvaluation) -> dict[str, Any]:
    def cell_dict(c: CellCommDiagnosis) -> dict[str, Any]:
        return {
            "label": c.label,
            "model_key": c.model_key,
            "concurrency": c.concurrency,
            "measured": {
                "tp1_tok_s": c.measured_tp1_tok_s,
                "tp2_tok_s": c.measured_tp2_tok_s,
                "delta_percent": c.measured_delta_percent,
                "tp1_ms_per_tok": c.measured_tp1_ms_per_tok,
                "tp2_ms_per_tok": c.measured_tp2_ms_per_tok,
                "excess_ms_per_tok": c.measured_excess_ms_per_tok,
                "regime": c.measured_regime,
            },
            "architecture_assumptions": {
                "num_layers": c.assumed_num_layers,
                "collectives_per_token": c.assumed_collectives_per_token,
            },
            "derived_proxy": {
                "observed_excess_us_per_collective_proxy": c.observed_excess_us_per_collective_proxy,
                "caveat": c.proxy_caveat,
            },
            "hypothetical": {
                "enabled": c.hypothetical_enabled,
                "t_comm_ms_per_tok": c.hypothetical_t_comm_ms_per_tok,
                "residual_ms_per_tok": c.hypothetical_residual_ms_per_tok,
                "note": c.hypothetical_note,
            },
            "summary": c.summary,
        }

    out: dict[str, Any] = {
        "milestone": ev.milestone,
        "notes": list(ev.notes),
        "cells": [cell_dict(c) for c in ev.cells],
        "crossover": asdict(ev.crossover) if ev.crossover else None,
    }
    return out
