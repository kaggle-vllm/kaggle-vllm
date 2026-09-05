"""Unit tests for Milestone 3 communication-cost diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kaggle_vllm.diagnostics.alpha_beta_model import (
    AlphaBetaCommModel,
    HypotheticalAlphaBetaAssumptions,
    ModelArchitectureSpecs,
    UnsupportedModelArchitectureError,
)
from kaggle_vllm.diagnostics.evaluator import (
    EvidenceIncompleteError,
    MilestoneArtifactEvaluator,
)
from kaggle_vllm.diagnostics.report import generate_reports
from kaggle_vllm.diagnostics.report import main as diagnostics_main

# ---------------------------------------------------------------------------
# Architecture / collective counts
# ---------------------------------------------------------------------------


def test_opt125m_layer_and_collective_counts() -> None:
    m = AlphaBetaCommModel()
    sp = m.get_specs("opt-125m")
    assert sp.num_layers == 12
    assert sp.assumed_allreduce_calls_per_token == 24


def test_qwen_layer_and_collective_counts() -> None:
    m = AlphaBetaCommModel()
    sp = m.get_specs("qwen2.5-3b")
    assert sp.num_layers == 36
    assert sp.assumed_allreduce_calls_per_token == 72


def test_unknown_model_fails_closed() -> None:
    m = AlphaBetaCommModel()
    with pytest.raises(UnsupportedModelArchitectureError):
        m.get_specs("no-such-model-xyz")


def test_architecture_override_allows_custom_model() -> None:
    custom = ModelArchitectureSpecs(
        name="custom-1b",
        num_layers=10,
        hidden_size=1024,
        num_attention_heads=8,
    )
    m = AlphaBetaCommModel(architecture_overrides={"custom-1b": custom})
    sp = m.get_specs("custom-1b")
    assert sp.num_layers == 10
    assert sp.assumed_allreduce_calls_per_token == 20


# ---------------------------------------------------------------------------
# Excess-time proxy (NOT classical alpha)
# ---------------------------------------------------------------------------


def test_excess_proxy_matches_hand_calculation_graph_like_numbers() -> None:
    """Proxy = excess_ms/tok / N_collectives * 1000 us — not claimed as alpha."""
    m = AlphaBetaCommModel()
    tp1, tp2 = 1921.1689664766213, 1407.6017475190556
    d = m.diagnose_cell(
        label="t",
        model_key="opt-125m",
        tp1_tok_s=tp1,
        tp2_tok_s=tp2,
        concurrency=1,
    )
    t1 = 1000.0 / tp1
    t2 = 1000.0 / tp2
    excess_ms = t2 - t1
    expected_us = (excess_ms / 24.0) * 1000.0

    assert d.assumed_num_layers == 12
    assert d.assumed_collectives_per_token == 24
    assert d.measured_regime == "TP2_SLOWER"
    assert d.measured_excess_ms_per_tok == pytest.approx(excess_ms, rel=1e-12)
    assert d.measured_service_time_delta_ms_per_tok == pytest.approx(
        excess_ms, rel=1e-12
    )
    assert d.observed_excess_us_per_collective_proxy == pytest.approx(
        expected_us, rel=1e-9
    )
    assert "not" in d.proxy_caveat.lower() or "NOT" in d.proxy_caveat
    assert (
        "alpha" in d.proxy_caveat.lower()
        or "ALPHA" in d.proxy_caveat
        or "proxy" in d.proxy_caveat.lower()
    )


def test_eager_proxy_much_larger_than_graph_proxy() -> None:
    m = AlphaBetaCommModel()
    graph = m.diagnose_cell(
        label="g",
        model_key="opt-125m",
        tp1_tok_s=1921.1689664766213,
        tp2_tok_s=1407.6017475190556,
    )
    eager = m.diagnose_cell(
        label="e",
        model_key="opt-125m",
        tp1_tok_s=312.0794092620273,
        tp2_tok_s=172.3194758116943,
    )
    assert graph.observed_excess_us_per_collective_proxy is not None
    assert eager.observed_excess_us_per_collective_proxy is not None
    assert eager.observed_excess_us_per_collective_proxy > (
        5 * graph.observed_excess_us_per_collective_proxy
    )


def test_tp2_faster_has_no_positive_collective_proxy() -> None:
    m = AlphaBetaCommModel()
    d = m.diagnose_cell(
        label="cross",
        model_key="qwen2.5-3b",
        tp1_tok_s=138.75,
        tp2_tok_s=174.27,
        concurrency=16,
    )
    assert d.measured_regime == "TP2_FASTER"
    assert d.observed_excess_us_per_collective_proxy is None
    assert d.measured_excess_ms_per_tok is None
    assert d.measured_service_time_delta_ms_per_tok is not None
    assert d.measured_service_time_delta_ms_per_tok < 0


def test_tp_equivalent_regime_with_tolerance() -> None:
    m = AlphaBetaCommModel()
    d = m.diagnose_cell(
        label="eq",
        model_key="opt-125m",
        tp1_tok_s=100.0,
        tp2_tok_s=100.0,
    )
    assert d.measured_regime == "TP_EQUIVALENT"
    assert d.observed_excess_us_per_collective_proxy is None


def test_invalid_throughput() -> None:
    m = AlphaBetaCommModel()
    d = m.diagnose_cell(
        label="x",
        model_key="opt-125m",
        tp1_tok_s=0.0,
        tp2_tok_s=10.0,
    )
    assert d.measured_regime == "INVALID"


def test_non_tp_comparison_skips_proxy() -> None:
    m = AlphaBetaCommModel()
    d = m.diagnose_cell(
        label="batching",
        model_key="qwen2.5-3b",
        tp1_tok_s=56.6,
        tp2_tok_s=55.0,
        comparison_is_tp_controlled=False,
    )
    assert d.observed_excess_us_per_collective_proxy is None
    assert "not a controlled tp1/tp2" in d.summary.lower()


# ---------------------------------------------------------------------------
# Hypothetical path: no baked-in 7.91 / 7.8
# ---------------------------------------------------------------------------


def test_enable_hypothetical_requires_explicit_assumptions() -> None:
    with pytest.raises(ValueError):
        AlphaBetaCommModel(enable_hypothetical=True, hypothetical=None)


def test_hypothetical_requires_source_note() -> None:
    with pytest.raises(ValueError):
        HypotheticalAlphaBetaAssumptions(
            alpha_us_per_collective=1.0,
            beta_gb_s=1.0,
            payload_mode="unknown",
            source_note="   ",
        )


def test_hypothetical_unknown_payload_does_not_use_concurrency_as_s() -> None:
    hyp = HypotheticalAlphaBetaAssumptions(
        alpha_us_per_collective=1.0,
        beta_gb_s=10.0,
        payload_mode="unknown",
        source_note="unit-test explicit assumption",
    )
    m = AlphaBetaCommModel(hypothetical=hyp, enable_hypothetical=True)
    a = m.diagnose_cell(
        label="a",
        model_key="opt-125m",
        tp1_tok_s=100.0,
        tp2_tok_s=50.0,
        concurrency=1,
    )
    b = m.diagnose_cell(
        label="b",
        model_key="opt-125m",
        tp1_tok_s=100.0,
        tp2_tok_s=50.0,
        concurrency=64,
    )
    assert a.hypothetical_t_comm_ms_per_tok is not None
    assert b.hypothetical_t_comm_ms_per_tok is not None
    assert a.hypothetical_t_comm_ms_per_tok == pytest.approx(
        b.hypothetical_t_comm_ms_per_tok
    )


# ---------------------------------------------------------------------------
# Evaluator: real M1/M2 artifacts
# ---------------------------------------------------------------------------


def test_m1_m2_parse_smoke() -> None:
    m1 = MilestoneArtifactEvaluator(
        "artifacts/kaggle-2026-09-01-milestone-1",
        strict=True,
    ).parse_m1_evidence()
    m2 = MilestoneArtifactEvaluator(
        "artifacts/kaggle-2026-09-02-milestone-2",
        strict=True,
    ).parse_m2_concurrency_matrix()

    assert any("opt125m-graph" in c.label for c in m1.cells)
    assert m1.errors == []
    assert len(m2.cells) == 6
    assert m2.errors == []
    assert m2.crossover is not None
    assert m2.crossover.measured_throughput_crossover_concurrency == 16
    assert m2.crossover.model_predicted_crossover_concurrency is None
    assert m2.crossover.prediction_status == "unsupported"


def test_m2_missing_dir_strict_raises(tmp_path: Path) -> None:
    ev = MilestoneArtifactEvaluator(tmp_path / "does-not-exist", strict=True)
    with pytest.raises(EvidenceIncompleteError):
        ev.parse_m2_concurrency_matrix()


def test_m2_missing_dir_nonstrict_records_errors(tmp_path: Path) -> None:
    ev = MilestoneArtifactEvaluator(tmp_path / "does-not-exist", strict=False)
    m2 = ev.parse_m2_concurrency_matrix()
    assert m2.cells == []
    assert m2.errors
    assert any("missing" in e.lower() for e in m2.errors)


def test_m1_malformed_json_strict(tmp_path: Path) -> None:
    bad = tmp_path / "comparison-broken.json"
    bad.write_text("{not-json", encoding="utf-8")
    ev = MilestoneArtifactEvaluator(tmp_path, strict=True)
    with pytest.raises(EvidenceIncompleteError):
        ev.parse_m1_evidence()


def test_evaluator_rejects_bare_enable_hypothetical_flag() -> None:
    with pytest.raises(ValueError):
        MilestoneArtifactEvaluator(
            "artifacts/kaggle-2026-09-02-milestone-2",
            enable_hypothetical=True,
        )


# ---------------------------------------------------------------------------
# Report / CLI
# ---------------------------------------------------------------------------


def test_report_ascii_and_no_bel(tmp_path: Path) -> None:
    paths = generate_reports(
        "artifacts/kaggle-2026-09-01-milestone-1",
        "artifacts/kaggle-2026-09-02-milestone-2",
        tmp_path,
        formats={"md", "json"},
        strict_evidence=True,
    )
    md = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "\x07" not in md
    assert "unsupported" in md.lower()
    assert "12" in md  # OPT layers called out in table/notes path
    assert "Provenance" in md or "provenance" in md.lower()

    js = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert js["m2"]["crossover"]["prediction_status"] == "unsupported"
    assert "provenance" in js
    assert js["provenance"].get("git_head")
    assert "markdown_sha256" in js["provenance"] or "json_sha256" in js["provenance"]


def test_format_md_only_writes_md_not_json(tmp_path: Path) -> None:
    paths = generate_reports(
        "artifacts/kaggle-2026-09-01-milestone-1",
        "artifacts/kaggle-2026-09-02-milestone-2",
        tmp_path,
        formats={"md"},
        strict_evidence=True,
    )
    assert "markdown" in paths
    assert "json" not in paths
    assert (tmp_path / "M3_DIAGNOSTICS_REPORT.md").is_file()
    assert not (tmp_path / "M3_DIAGNOSTICS_REPORT.json").exists()
    # provenance sidecar is still useful
    assert (tmp_path / "M3_PROVENANCE.json").is_file()


def test_format_json_only_writes_json_not_md(tmp_path: Path) -> None:
    paths = generate_reports(
        "artifacts/kaggle-2026-09-01-milestone-1",
        "artifacts/kaggle-2026-09-02-milestone-2",
        tmp_path,
        formats={"json"},
        strict_evidence=True,
    )
    assert "json" in paths
    assert "markdown" not in paths
    assert (tmp_path / "M3_DIAGNOSTICS_REPORT.json").is_file()
    assert not (tmp_path / "M3_DIAGNOSTICS_REPORT.md").exists()


def test_cli_main_runs(tmp_path: Path) -> None:
    rc = diagnostics_main(
        [
            "--m1-dir",
            "artifacts/kaggle-2026-09-01-milestone-1",
            "--m2-dir",
            "artifacts/kaggle-2026-09-02-milestone-2",
            "--output-dir",
            str(tmp_path),
            "--format",
            "both",
        ]
    )
    assert rc == 0
    assert (tmp_path / "M3_DIAGNOSTICS_REPORT.md").is_file()
    assert (tmp_path / "M3_DIAGNOSTICS_REPORT.json").is_file()
    assert (tmp_path / "M3_PROVENANCE.json").is_file()


def test_cli_format_md(tmp_path: Path) -> None:
    rc = diagnostics_main(
        [
            "--m1-dir",
            "artifacts/kaggle-2026-09-01-milestone-1",
            "--m2-dir",
            "artifacts/kaggle-2026-09-02-milestone-2",
            "--output-dir",
            str(tmp_path),
            "--format",
            "md",
        ]
    )
    assert rc == 0
    assert (tmp_path / "M3_DIAGNOSTICS_REPORT.md").is_file()
    assert not (tmp_path / "M3_DIAGNOSTICS_REPORT.json").exists()
