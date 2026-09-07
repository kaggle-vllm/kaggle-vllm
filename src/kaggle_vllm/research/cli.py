"""CLI for fail-closed CPU-side M3 evidence analysis."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .comparison import build_model_vs_observed, load_payload_scenarios
from .errors import ResearchEvidenceError
from .measured_comm import (
    assert_equivalent_ledgers,
    fit_communication_model,
    load_allreduce_csv,
    load_allreduce_json,
    summarize_allreduce,
)
from .provenance import verify_sha256_manifest
from .report import write_m3_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kaggle_vllm.research")
    subcommands = parser.add_subparsers(dest="command", required=True)
    analyze = subcommands.add_parser("analyze-m3")
    analyze.add_argument("--raw-csv", type=Path, required=True)
    analyze.add_argument("--raw-json", type=Path, required=True)
    analyze.add_argument("--m1-dir", type=Path, required=True)
    analyze.add_argument("--m2-dir", type=Path, required=True)
    analyze.add_argument("--payload-scenarios", type=Path)
    analyze.add_argument("--output-dir", type=Path, required=True)
    verify = subcommands.add_parser("verify-hashes")
    verify.add_argument("root", type=Path)
    verify.add_argument("manifest", type=Path, nargs="?")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-hashes":
            manifest = args.manifest or args.root / "SHA256SUMS.txt"
            verified = verify_sha256_manifest(args.root, manifest)
            print(json.dumps({"status": "verified", "files": verified}, indent=2))
            return 0
        csv_rows = load_allreduce_csv(args.raw_csv)
        json_rows = load_allreduce_json(args.raw_json)
        assert_equivalent_ledgers(csv_rows, json_rows)
        summaries = summarize_allreduce(csv_rows)
        fit = fit_communication_model(summaries)
        scenarios = load_payload_scenarios(args.payload_scenarios)
        comparison = build_model_vs_observed(
            args.m1_dir, args.m2_dir, fit=fit, payload_scenarios=scenarios
        )
        written = write_m3_outputs(
            args.output_dir,
            summaries=summaries,
            fit=fit,
            comparison=comparison,
        )
    except ResearchEvidenceError as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": "MEASURED_ON_KAGGLE_PENDING_REVIEW",
                "written": [str(path) for path in written],
            },
            indent=2,
        )
    )
    return 0
