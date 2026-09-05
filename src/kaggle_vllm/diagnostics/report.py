"""Markdown/JSON report writer for Milestone 3 diagnostics."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fmt(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return bool(out)
    except (OSError, subprocess.CalledProcessError):
        return True


def render_markdown(m1: Any, m2: Any, provenance: dict[str, Any]) -> str:
    gen_utc = provenance.get("generated_at_utc")
    g_head = provenance.get("git_head")
    g_dirty = provenance.get("git_tree_dirty")
    cmd = provenance.get("command")
    hyp = provenance.get("enable_hypothetical")
    strict = provenance.get("strict_evidence")
    m1_d = provenance.get("m1_dir")
    m2_d = provenance.get("m2_dir")
    out_d = provenance.get("output_dir")
    fmts = provenance.get("formats_written")

    lines: list[str] = [
        "# Milestone 3 - Communication-Cost Diagnostics Report",
        "",
        "## Provenance",
        "",
        f"- Generated at (UTC): {gen_utc}",
        f"- Generator git HEAD: `{g_head}`",
        f"- Generator git dirty: `{g_dirty}`",
        f"- Command: `{cmd}`",
        f"- enable_hypothetical: {hyp}",
        f"- strict_evidence: {strict}",
        f"- M1 dir: `{m1_d}`",
        f"- M2 dir: `{m2_d}`",
        f"- Output dir: `{out_d}`",
        f"- Formats written: {fmts}",
        "",
        "## Scientific boundaries",
        "",
        "1. Measured quantities come from M1/M2 JSON evidence.",
        "2. Derived proxy = positive excess service time / assumed collectives; NOT isolated alpha.",
        "3. Hypothetical alpha-beta requires explicit caller parameters (no baked-in 7.91/7.8).",
        "4. PHB topology may be observed; it is not sole NCCL/PCIe causality.",
        "",
        "## 1. Milestone 1",
        "",
    ]
    if m1.errors:
        lines.append("### Evidence errors")
        lines.extend([f"- ERROR: {e}" for e in m1.errors])
        lines.append("")
    lines.extend([f"- {n}" for n in m1.notes])
    lines += [
        "",
        "| Comparison | TP1 tok/s | TP2 tok/s | Delta % | Layers | Collectives/tok | Proxy us/coll | Signed dt ms/tok | Regime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in m1.cells:
        lines.append(
            f"| `{c.label}` | {_fmt(c.measured_tp1_tok_s, 1)} | {_fmt(c.measured_tp2_tok_s, 1)} | "
            f"{c.measured_delta_percent:+.1f} | {c.assumed_num_layers} | {c.assumed_collectives_per_token} | "
            f"{_fmt(c.observed_excess_us_per_collective_proxy)} | "
            f"{_fmt(c.measured_service_time_delta_ms_per_tok)} | {c.measured_regime} |"
        )
    lines.append("")
    for c in m1.cells:
        lines.append(f"- `{c.label}`: {c.summary}")
    lines += ["", "## 2. Milestone 2", ""]
    if m2.errors:
        lines.append("### Evidence errors")
        lines.extend([f"- ERROR: {e}" for e in m2.errors])
        lines.append("")
    lines.extend([f"- {n}" for n in m2.notes])
    lines += [
        "",
        "| c | TP1 tok/s | TP2 tok/s | Delta % | Excess ms/tok | Signed dt ms/tok | Proxy us/coll | Regime |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in m2.cells:
        lines.append(
            f"| {c.concurrency} | {_fmt(c.measured_tp1_tok_s)} | {_fmt(c.measured_tp2_tok_s)} | "
            f"{c.measured_delta_percent:+.1f} | {_fmt(c.measured_excess_ms_per_tok)} | "
            f"{_fmt(c.measured_service_time_delta_ms_per_tok)} | "
            f"{_fmt(c.observed_excess_us_per_collective_proxy)} | {c.measured_regime} |"
        )
    lines += ["", "## 3. Crossover: measured vs model prediction", ""]
    if m2.crossover is None:
        lines.append("No crossover object.")
    else:
        x = m2.crossover
        lines.append(
            f"- Measured throughput crossover concurrency: {x.measured_throughput_crossover_concurrency}"
        )
        lines.append(
            f"- Measured rates: {_fmt(x.measured_crossover_tp1_tok_s)} -> {_fmt(x.measured_crossover_tp2_tok_s)} tok/s"
        )
        lines.append(
            f"- Model-predicted crossover concurrency: {x.model_predicted_crossover_concurrency}"
        )
        lines.append(f"- Prediction status: `{x.prediction_status}`")
        lines.append(f"- Reason: {x.prediction_reason}")
    lines += [
        "",
        "## 4. Limitations",
        "",
        "- Eager vs graph proxy divergence means proxy is not stable transport alpha.",
        "- Serving concurrency is not silently treated as decode batch size.",
        "- No sole NCCL/PCIe causality claim from M1/M2 alone.",
        "",
    ]
    return "\n".join(lines) + "\n"


def generate_reports(
    m1_dir: str | Path,
    m2_dir: str | Path,
    output_dir: str | Path,
    *,
    enable_hypothetical: bool = False,
    formats: set[str] | None = None,
    strict_evidence: bool = True,
    command: str = "python -m kaggle_vllm.diagnostics",
) -> dict[str, str]:
    from kaggle_vllm.diagnostics.evaluator import (
        MilestoneArtifactEvaluator,
        evaluation_to_jsonable,
    )

    if enable_hypothetical:
        raise ValueError(
            "Pass explicit HypotheticalAlphaBetaAssumptions via API; CLI flag alone is disabled"
        )

    formats = formats or {"md", "json"}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    m1_eval = MilestoneArtifactEvaluator(m1_dir, strict=strict_evidence)
    m2_eval = MilestoneArtifactEvaluator(m2_dir, strict=strict_evidence)
    m1 = m1_eval.parse_m1_evidence()
    m2 = m2_eval.parse_m2_concurrency_matrix()

    provenance: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "git_tree_dirty": _git_dirty(),
        "command": command,
        "enable_hypothetical": False,
        "strict_evidence": strict_evidence,
        "m1_dir": str(m1_dir),
        "m2_dir": str(m2_dir),
        "output_dir": str(output_path),
        "formats_written": sorted(formats),
        "errors": list(m1.errors) + list(m2.errors),
    }

    paths: dict[str, str] = {}

    if "md" in formats:
        md_path = output_path / "M3_DIAGNOSTICS_REPORT.md"
        md_text = render_markdown(m1, m2, provenance)
        md_path.write_text(md_text, encoding="utf-8")
        provenance["markdown_sha256"] = _sha256_file(md_path)
        paths["markdown"] = str(md_path)

    if "json" in formats:
        json_path = output_path / "M3_DIAGNOSTICS_REPORT.json"
        payload = {
            "provenance": dict(provenance),
            "m1": evaluation_to_jsonable(m1),
            "m2": evaluation_to_jsonable(m2),
        }
        json_bytes = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
        json_path.write_bytes(json_bytes)
        provenance["json_sha256"] = _sha256_file(json_path)
        paths["json"] = str(json_path)

    prov_path = output_path / "M3_PROVENANCE.json"
    prov_bytes = (json.dumps(provenance, indent=2) + "\n").encode("utf-8")
    prov_path.write_bytes(prov_bytes)
    paths["provenance"] = str(prov_path)

    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Milestone 3 communication-cost diagnostics"
    )
    p.add_argument("--m1-dir", default="artifacts/kaggle-2026-09-01-milestone-1")
    p.add_argument("--m2-dir", default="artifacts/kaggle-2026-09-02-milestone-2")
    p.add_argument("--output-dir", default="artifacts/kaggle-2026-09-03-milestone-3")
    p.add_argument("--format", choices=("md", "json", "both"), default="both")
    p.add_argument("--no-strict-evidence", action="store_true")
    args = p.parse_args(argv)

    formats = {"md", "json"} if args.format == "both" else {args.format}
    cmd = " ".join(["python -m kaggle_vllm.diagnostics", *(argv or sys.argv[1:])])
    try:
        paths = generate_reports(
            args.m1_dir,
            args.m2_dir,
            args.output_dir,
            formats=formats,
            strict_evidence=not args.no_strict_evidence,
            command=cmd,
        )
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    for key in ("markdown", "json", "provenance"):
        if key in paths:
            print(paths[key])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
