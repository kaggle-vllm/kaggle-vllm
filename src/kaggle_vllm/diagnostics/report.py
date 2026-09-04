"""
Report Generator for Communication Cost Diagnostics.

Consumes evaluation results from MilestoneArtifactEvaluator and produces
a formatted Markdown artifact summarizing M1/M2 communication overheads.
"""

from pathlib import Path
from typing import Optional
from kaggle_vllm.diagnostics.evaluator import MilestoneArtifactEvaluator


def generate_markdown_report(m1_dir: str, m2_dir: str, output_path: Optional[str] = None) -> str:
    m1_eval = MilestoneArtifactEvaluator(m1_dir)
    m1_results = m1_eval.parse_m1_evidence()

    m2_eval = MilestoneArtifactEvaluator(m2_dir)
    m2_results = m2_eval.parse_m2_concurrency_matrix()

    md = []
    md.append("# Milestone 3 — Communication-Cost & Crossover Diagnostics Report\n")
    md.append("**Framework:** `kaggle-vllm` Analytical Alpha-Beta Diagnostics  ")
    md.append("**Scope:** Evaluation of PCIe Host Bridge (PHB) AllReduce penalty across M1 and M2 datasets\n")

    md.append("## 1. Milestone 1 — Low-Concurrency TP Penalty Analysis\n")
    md.append("| Comparison | TP1 (tok/s) | TP2 (tok/s) | Delta (%) | Inferred Step Alpha (µs) | Diagnostic Note |")
    md.append("|---|---|---|---|---|---|")

    for label, res in m1_results.items():
        est = res["estimate"]
        alpha_str = f"{est.inferred_alpha_us_per_step:.2f} µs" if est.inferred_alpha_us_per_step > 0 else "N/A"
        md.append(f"| `{label}` | {est.measured_tp1_tok_s:.1f} | {est.measured_tp2_tok_s:.1f} | {est.observed_delta_percent:+.1f}% | {alpha_str} | {est.explanation} |")

    md.append("\n## 2. Milestone 2 — Concurrency Crossover Matrix\n")
    md.append("| Concurrency (c) | TP1 tok/s | TP2 tok/s | Delta (%) | TP1 TTFT p95 | TP2 TTFT p95 | System Regime |")
    md.append("|---|---|---|---|---|---|---|")

    for cell in m2_results:
        est = cell.comm_estimate
        regime = "**COMMUNICATION-TAX**" if est.is_comm_bound else "**TP2-ADVANTAGE**"
        ttft1 = f"{cell.tp1_ttft_p95_s:.2f}s" if cell.tp1_ttft_p95_s is not None else "n/a"
        ttft2 = f"{cell.tp2_ttft_p95_s:.2f}s" if cell.tp2_ttft_p95_s is not None else "n/a"
        md.append(f"| {cell.concurrency} | {cell.tp1_throughput_tok_s:.2f} | {cell.tp2_throughput_tok_s:.2f} | {est.observed_delta_percent:+.1f}% | {ttft1} | {ttft2} | {regime} |")

    cross = next((c for c in m2_results if c.tp2_throughput_tok_s > c.tp1_throughput_tok_s), None)
    if cross:
        md.append(f"\n> **Observed Throughput Crossover Point:** Concurrency **c = {cross.concurrency}** ({cross.tp1_throughput_tok_s:.2f} → {cross.tp2_throughput_tok_s:.2f} tok/s).\n")

    md.append("## 3. Causal Boundaries & Limitations\n")
    md.append("1. **Measured vs. Inferred:** Throughput and latencies are measured quantities from JSON evidence. Step alpha ($\alpha$) and bus bandwidth ($\beta$) are inferred analytical parameters.")
    md.append("2. **Non-Sole Causality:** While PCIe Host Bridge (PHB) latency explains the low-concurrency TP2 penalty, overall performance reflects interacting effects of scheduling, CUDA graph capture, memory bandwidth, and vLLM batching.")
    md.append("3. **Scope:** Results apply specifically to the pinned Kaggle Dual-T4 (SM75) environment and should not be extrapolated to NVLink or multi-node interconnects.")

    report_content = "\n".join(md)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report_content, encoding="utf-8")
        print(f"Report saved to: {output_path}")

    return report_content


if __name__ == "__main__":
    generate_markdown_report(
        "artifacts/kaggle-2026-09-01-milestone-1",
        "artifacts/kaggle-2026-09-02-milestone-2",
        "artifacts/kaggle-2026-09-03-milestone-3/M3_DIAGNOSTICS_REPORT.md"
    )
