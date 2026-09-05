"""Load M1/M2 evidence and attach communication-cost diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kaggle_vllm.diagnostics.alpha_beta_model import (
    AlphaBetaCommModel,
    CellCommDiagnosis,
    UnsupportedModelArchitectureError,
)


class EvidenceIncompleteError(RuntimeError):
    """Required milestone evidence missing or malformed."""


@dataclass(frozen=True)
class CrossoverAnalysis:
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
    errors: list[str]


class MilestoneArtifactEvaluator:
    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        enable_hypothetical: bool = False,
        strict: bool = True,
    ) -> None:
        if enable_hypothetical:
            raise ValueError(
                "enable_hypothetical requires wiring HypotheticalAlphaBetaAssumptions "
                "explicitly in a future call path; refusing bare True without assumptions"
            )
        self.artifact_path = Path(artifact_dir)
        self.model = AlphaBetaCommModel(enable_hypothetical=False)
        self.strict = strict

    def _fail_or_note(self, errors: list[str], msg: str) -> None:
        errors.append(msg)

    def parse_m1_evidence(self) -> MilestoneEvaluation:
        cells: list[CellCommDiagnosis] = []
        notes = [
            "M1 offline comparisons: treat graph vs eager proxy spread as a stability warning.",
            "M1 text already states communication was not causally isolated.",
        ]
        errors: list[str] = []

        if not self.artifact_path.is_dir():
            errors.append(f"M1 artifact dir missing: {self.artifact_path}")
            return self._finish("m1", cells, None, notes, errors)

        comp_files = sorted(self.artifact_path.glob("comparison-*.json"))
        if not comp_files:
            errors.append(f"No comparison-*.json under {self.artifact_path}")
            return self._finish("m1", cells, None, notes, errors)

        for comp_file in comp_files:
            try:
                data = json.loads(comp_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeError) as exc:
                errors.append(f"unreadable {comp_file.name}: {exc}")
                continue

            label = comp_file.stem
            abs_data = data.get("absolute")
            if not isinstance(abs_data, dict):
                errors.append(f"{comp_file.name}: missing absolute block")
                continue
            try:
                if "baseline_output_tokens_per_second" not in abs_data:
                    errors.append(
                        f"{comp_file.name}: missing baseline_output_tokens_per_second"
                    )
                    continue
                if "candidate_output_tokens_per_second" not in abs_data:
                    errors.append(
                        f"{comp_file.name}: missing candidate_output_tokens_per_second"
                    )
                    continue
                tp1 = float(abs_data["baseline_output_tokens_per_second"])
                tp2 = float(abs_data["candidate_output_tokens_per_second"])
            except (TypeError, ValueError) as exc:
                errors.append(f"{comp_file.name}: bad throughput fields: {exc}")
                continue

            is_tp = "batching" not in label
            label_l = label.lower()
            if "opt125m" in label_l or "opt-125m" in label_l:
                model_key = "opt-125m"
            elif "qwen" in label_l:
                model_key = "qwen2.5-3b"
            else:
                errors.append(
                    f"{comp_file.name}: cannot map label to a known model_key "
                    f"(expected opt125m/opt-125m or qwen); refusing silent fallback"
                )
                continue
            try:
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
            except UnsupportedModelArchitectureError as exc:
                errors.append(str(exc))

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
                f"OPT-125M excess-time proxy differs: graph~{graph_proxy:.2f} vs "
                f"eager~{eager_proxy:.2f} us/collective (not stable transport alpha)."
            )

        return self._finish("m1", cells, None, notes, errors)

    def parse_m2_concurrency_matrix(self) -> MilestoneEvaluation:
        notes = [
            "M2 online serving: concurrency is NOT proven equal to instantaneous decode batch.",
            "Crossover c=16 is MEASURED detection from the matrix, not an alpha-beta prediction.",
        ]
        errors: list[str] = []
        cells: list[CellCommDiagnosis] = []

        if not self.artifact_path.is_dir():
            errors.append(f"M2 artifact dir missing: {self.artifact_path}")
            return self._finish("m2", cells, None, notes, errors)

        summary_path = self.artifact_path / "summary.json"
        if not summary_path.exists():
            errors.append("summary.json missing")
            return self._finish("m2", cells, None, notes, errors)

        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            errors.append(f"summary.json unreadable: {exc}")
            return self._finish("m2", cells, None, notes, errors)

        matrix = data.get("matrix")
        if not isinstance(matrix, list) or not matrix:
            errors.append("matrix missing or empty in summary.json")
            return self._finish("m2", cells, None, notes, errors)

        by_c: dict[int, dict[int, dict]] = {}
        for idx, row in enumerate(matrix):
            if not isinstance(row, dict):
                errors.append(f"matrix[{idx}] is not an object")
                continue
            try:
                c = int(row["concurrency"])
                tp = int(row["tensor_parallel_size"])
                thr = float(row["output_throughput_tokens_per_second"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"matrix[{idx}] invalid fields: {exc}")
                continue
            row = dict(row)
            row["output_throughput_tokens_per_second"] = thr
            by_c.setdefault(c, {})[tp] = row

        expected_c = [1, 4, 8, 16, 32, 64]
        for c in expected_c:
            if c not in by_c or 1 not in by_c[c] or 2 not in by_c[c]:
                errors.append(f"incomplete TP1/TP2 pair at concurrency={c}")
                continue
            r1, r2 = by_c[c][1], by_c[c][2]
            tp1 = float(r1["output_throughput_tokens_per_second"])
            tp2 = float(r2["output_throughput_tokens_per_second"])
            try:
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
            except UnsupportedModelArchitectureError as exc:
                errors.append(str(exc))

        measured_cross = None
        t1 = t2 = None
        for cell in cells:
            if cell.measured_regime == "TP2_FASTER":
                measured_cross = cell.concurrency
                t1, t2 = cell.measured_tp1_tok_s, cell.measured_tp2_tok_s
                break

        analysis = (
            data.get("analysis") if isinstance(data.get("analysis"), dict) else {}
        )
        declared = analysis.get("throughput_crossover_concurrency")
        if declared is not None and measured_cross is not None:
            try:
                if int(declared) != measured_cross:
                    notes.append(
                        f"summary.analysis crossover={declared} differs from recomputed={measured_cross}."
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"invalid analysis.throughput_crossover_concurrency={declared!r}"
                )

        crossover = CrossoverAnalysis(
            measured_throughput_crossover_concurrency=measured_cross,
            measured_crossover_tp1_tok_s=t1,
            measured_crossover_tp2_tok_s=t2,
            model_predicted_crossover_concurrency=None,
            prediction_status="unsupported",
            prediction_reason=(
                "M1/M2 lack isolated collective timestamps and trusted per-step "
                "scheduler batch sizes; alpha-beta cannot defensibly predict crossover."
            ),
        )
        return self._finish("m2", cells, crossover, notes, errors)

    def _finish(
        self,
        milestone: str,
        cells: list[CellCommDiagnosis],
        crossover: CrossoverAnalysis | None,
        notes: list[str],
        errors: list[str],
    ) -> MilestoneEvaluation:
        ev = MilestoneEvaluation(milestone, cells, crossover, notes, errors)
        if self.strict and errors:
            raise EvidenceIncompleteError(
                f"{milestone} evidence incomplete: " + "; ".join(errors)
            )
        return ev


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
                "service_time_delta_ms_per_tok": c.measured_service_time_delta_ms_per_tok,
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

    return {
        "milestone": ev.milestone,
        "notes": list(ev.notes),
        "errors": list(ev.errors),
        "cells": [cell_dict(c) for c in ev.cells],
        "crossover": asdict(ev.crossover) if ev.crossover else None,
    }
