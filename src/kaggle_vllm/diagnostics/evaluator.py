"""
Evidence Evaluator for Milestone 1 & Milestone 2 Artifacts.

Parses JSON evidence, extracts measured throughput/latencies, and applies
AlphaBetaCommModel. Distinguishes measured vs inferred quantities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from kaggle_vllm.diagnostics.alpha_beta_model import AlphaBetaCommModel, CommCostEstimate


def _find_number(obj: Any, key_substrs: List[str], prefer_keys: Optional[List[str]] = None) -> Optional[float]:
    """Depth-first search for a numeric field whose key matches substrings."""
    prefer_keys = prefer_keys or []

    def key_ok(k: str) -> bool:
        kl = k.lower()
        return all(s.lower() in kl for s in key_substrs)

    # Prefer exact-ish keys first on dicts
    if isinstance(obj, dict):
        for pk in prefer_keys:
            if pk in obj and isinstance(obj[pk], (int, float)):
                return float(obj[pk])
        for k, v in obj.items():
            if key_ok(k) and isinstance(v, (int, float)):
                return float(v)
        for v in obj.values():
            r = _find_number(v, key_substrs, prefer_keys)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for item in obj:
            r = _find_number(item, key_substrs, prefer_keys)
            if r is not None:
                return r
    return None


class CellEvaluationResult:
    def __init__(
        self,
        concurrency: int,
        tp1_throughput_tok_s: float,
        tp2_throughput_tok_s: float,
        tp1_ttft_p95_s: Optional[float],
        tp2_ttft_p95_s: Optional[float],
        tp1_tpot_p95_s: Optional[float],
        tp2_tpot_p95_s: Optional[float],
        comm_estimate: CommCostEstimate,
    ):
        self.concurrency = concurrency
        self.tp1_throughput_tok_s = tp1_throughput_tok_s
        self.tp2_throughput_tok_s = tp2_throughput_tok_s
        self.tp1_ttft_p95_s = tp1_ttft_p95_s
        self.tp2_ttft_p95_s = tp2_ttft_p95_s
        self.tp1_tpot_p95_s = tp1_tpot_p95_s
        self.tp2_tpot_p95_s = tp2_tpot_p95_s
        self.comm_estimate = comm_estimate


class MilestoneArtifactEvaluator:
    def __init__(self, artifact_dir: str):
        self.artifact_path = Path(artifact_dir)
        self.comm_model = AlphaBetaCommModel()

    def parse_m1_evidence(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for comp_file in sorted(self.artifact_path.glob("comparison-*.json")):
            try:
                data = json.loads(comp_file.read_text(encoding="utf-8"))
                label = comp_file.stem
                abs_data = data.get("absolute", {})

                tp1_tok_s = float(abs_data.get("baseline_output_tokens_per_second", 0.0) or 0.0)
                tp2_tok_s = float(abs_data.get("candidate_output_tokens_per_second", 0.0) or 0.0)

                # Qwen batching comparison is NOT TP1 vs TP2 — skip causal TP framing
                if "qwen-tp2-batching" in label or "batching" in label:
                    model_key = "qwen2.5-3b"
                    # Still run model for numbers, but tag carefully
                    estimate = self.comm_model.evaluate_cell(
                        model_key=model_key,
                        tp1_tok_s=tp1_tok_s,
                        tp2_tok_s=tp2_tok_s,
                        concurrency=1,
                    )
                    # Override explanation: not a TP experiment
                    estimate = CommCostEstimate(
                        measured_tp1_tok_s=estimate.measured_tp1_tok_s,
                        measured_tp2_tok_s=estimate.measured_tp2_tok_s,
                        observed_delta_percent=estimate.observed_delta_percent,
                        inferred_alpha_us_per_step=0.0,
                        inferred_total_comm_latency_ms_per_tok=0.0,
                        is_comm_bound=False,
                        explanation=(
                            f"MEASURED only: baseline vs max_num_batched_tokens variant "
                            f"(delta {estimate.observed_delta_percent:+.1f}%). "
                            f"Not a controlled TP1/TP2 comparison — do not infer PHB AllReduce alpha."
                        ),
                    )
                else:
                    model_key = "opt-125m" if "opt125m" in label else "qwen2.5-3b"
                    estimate = self.comm_model.evaluate_cell(
                        model_key=model_key,
                        tp1_tok_s=tp1_tok_s,
                        tp2_tok_s=tp2_tok_s,
                        concurrency=1,
                    )

                results[label] = {
                    "comparison_label": label,
                    "tp1_tok_s": tp1_tok_s,
                    "tp2_tok_s": tp2_tok_s,
                    "estimate": estimate,
                }
            except Exception:
                continue
        return results

    def _load_m2_from_summary(self) -> Optional[List[CellEvaluationResult]]:
        summary_path = self.artifact_path / "summary.json"
        if not summary_path.exists():
            return None

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        matrix = data.get("matrix")
        if not isinstance(matrix, list) or not matrix:
            return None

        # Group by concurrency
        by_c: Dict[int, Dict[int, dict]] = {}
        for row in matrix:
            if not isinstance(row, dict):
                continue
            c = int(row.get("concurrency", -1))
            tp = int(row.get("tensor_parallel_size", -1))
            if c < 0 or tp < 0:
                continue
            by_c.setdefault(c, {})[tp] = row

        evaluations: List[CellEvaluationResult] = []
        for c in sorted(by_c.keys()):
            r1 = by_c[c].get(1)
            r2 = by_c[c].get(2)
            if not r1 or not r2:
                continue

            tp1_tok = float(r1.get("output_throughput_tokens_per_second", 0.0) or 0.0)
            tp2_tok = float(r2.get("output_throughput_tokens_per_second", 0.0) or 0.0)
            tp1_ttft = r1.get("ttft_p95_seconds")
            tp2_ttft = r2.get("ttft_p95_seconds")
            tp1_tpot = r1.get("tpot_p95_seconds")
            tp2_tpot = r2.get("tpot_p95_seconds")

            estimate = self.comm_model.evaluate_cell(
                model_key="qwen2.5-3b",
                tp1_tok_s=tp1_tok,
                tp2_tok_s=tp2_tok,
                concurrency=c,
            )
            evaluations.append(
                CellEvaluationResult(
                    concurrency=c,
                    tp1_throughput_tok_s=tp1_tok,
                    tp2_throughput_tok_s=tp2_tok,
                    tp1_ttft_p95_s=float(tp1_ttft) if tp1_ttft is not None else None,
                    tp2_ttft_p95_s=float(tp2_ttft) if tp2_ttft is not None else None,
                    tp1_tpot_p95_s=float(tp1_tpot) if tp1_tpot is not None else None,
                    tp2_tpot_p95_s=float(tp2_tpot) if tp2_tpot is not None else None,
                    comm_estimate=estimate,
                )
            )
        return evaluations

    def _extract_cell_metrics(self, data: dict) -> Dict[str, Optional[float]]:
        # Prefer known keys, then fuzzy search
        thr = _find_number(
            data,
            ["throughput"],
            prefer_keys=[
                "output_throughput_tokens_per_second",
                "output_throughput_tok_s",
                "output_tokens_per_second",
            ],
        )
        # If fuzzy matched wrong thing, try harder
        if thr is None or thr == 0.0:
            thr = _find_number(data, ["output", "throughput"])
        if thr is None:
            thr = _find_number(data, ["tokens_per_second"])

        ttft = _find_number(
            data,
            ["ttft"],
            prefer_keys=["ttft_p95_seconds", "ttft_p95_s"],
        )
        tpot = _find_number(
            data,
            ["tpot"],
            prefer_keys=["tpot_p95_seconds", "tpot_p95_s"],
        )
        return {"thr": thr, "ttft": ttft, "tpot": tpot}

    def parse_m2_concurrency_matrix(self) -> List[CellEvaluationResult]:
        # 1) Authoritative: summary.json
        from_summary = self._load_m2_from_summary()
        if from_summary:
            return from_summary

        # 2) Fallback: per-cell JSON files
        evaluations: List[CellEvaluationResult] = []
        for c in (1, 4, 8, 16, 32, 64):
            tp1_file = self.artifact_path / f"qwen-tp1-c{c:02d}.json"
            tp2_file = self.artifact_path / f"qwen-tp2-c{c:02d}.json"
            if not tp1_file.exists() or not tp2_file.exists():
                continue
            try:
                tp1_data = json.loads(tp1_file.read_text(encoding="utf-8"))
                tp2_data = json.loads(tp2_file.read_text(encoding="utf-8"))
                m1 = self._extract_cell_metrics(tp1_data)
                m2 = self._extract_cell_metrics(tp2_data)
                tp1_tok = float(m1["thr"] or 0.0)
                tp2_tok = float(m2["thr"] or 0.0)

                estimate = self.comm_model.evaluate_cell(
                    model_key="qwen2.5-3b",
                    tp1_tok_s=tp1_tok,
                    tp2_tok_s=tp2_tok,
                    concurrency=c,
                )
                evaluations.append(
                    CellEvaluationResult(
                        concurrency=c,
                        tp1_throughput_tok_s=tp1_tok,
                        tp2_throughput_tok_s=tp2_tok,
                        tp1_ttft_p95_s=m1["ttft"],
                        tp2_ttft_p95_s=m2["ttft"],
                        tp1_tpot_p95_s=m1["tpot"],
                        tp2_tpot_p95_s=m2["tpot"],
                        comm_estimate=estimate,
                    )
                )
            except Exception:
                continue
        return evaluations


def run_quick_evaluation() -> None:
    m1_eval = MilestoneArtifactEvaluator("artifacts/kaggle-2026-09-01-milestone-1")
    m1_results = m1_eval.parse_m1_evidence()

    m2_eval = MilestoneArtifactEvaluator("artifacts/kaggle-2026-09-02-milestone-2")
    m2_results = m2_eval.parse_m2_concurrency_matrix()

    print("=" * 90)
    print("      MILESTONE 3: DIAGNOSTIC EVALUATION OF M1 AND M2 ARTIFACTS")
    print("=" * 90)

    print("\n--- M1 Comparison Diagnostics ---")
    for label, res in m1_results.items():
        est = res["estimate"]
        print(f"[{label}]")
        print(
            f"  Measured: TP1/baseline={est.measured_tp1_tok_s:.1f} tok/s | "
            f"TP2/candidate={est.measured_tp2_tok_s:.1f} tok/s "
            f"(Delta: {est.observed_delta_percent:+.1f}%)"
        )
        if est.inferred_alpha_us_per_step > 0:
            print(f"  Inferred alpha (µs/step): {est.inferred_alpha_us_per_step:.2f}")
        print(f"  Note: {est.explanation}\n")

    print("--- M2 Concurrency Matrix Diagnostics (from summary.json when present) ---")
    print(
        f"{'c':>4} | {'TP1 tok/s':>10} | {'TP2 tok/s':>10} | {'Δ%':>8} | "
        f"{'TTFT p95 TP1':>12} | {'TTFT p95 TP2':>12} | regime"
    )
    print("-" * 90)
    for cell in m2_results:
        est = cell.comm_estimate
        regime = "COMM-TAX" if est.is_comm_bound else "TP2-WINS"
        ttft1 = f"{cell.tp1_ttft_p95_s:.2f}s" if cell.tp1_ttft_p95_s is not None else "n/a"
        ttft2 = f"{cell.tp2_ttft_p95_s:.2f}s" if cell.tp2_ttft_p95_s is not None else "n/a"
        print(
            f"{cell.concurrency:>4} | {cell.tp1_throughput_tok_s:>10.2f} | "
            f"{cell.tp2_throughput_tok_s:>10.2f} | {est.observed_delta_percent:>+7.1f}% | "
            f"{ttft1:>12} | {ttft2:>12} | {regime}"
        )
    print("=" * 90)

    # Crossover callout
    cross = next((c for c in m2_results if c.tp2_throughput_tok_s > c.tp1_throughput_tok_s), None)
    if cross:
        print(
            f"\nFirst throughput crossover (measured): c={cross.concurrency} "
            f"({cross.tp1_throughput_tok_s:.2f} → {cross.tp2_throughput_tok_s:.2f} tok/s)"
        )
    print()


if __name__ == "__main__":
    run_quick_evaluation()
