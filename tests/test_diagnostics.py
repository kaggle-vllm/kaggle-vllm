import json
from pathlib import Path

import pytest

from kaggle_vllm.diagnostics.alpha_beta_model import (
    AlphaBetaCommModel,
    HypotheticalAlphaBetaAssumptions,
)
from kaggle_vllm.diagnostics.evaluator import MilestoneArtifactEvaluator
from kaggle_vllm.diagnostics.report import generate_reports


def test_opt125m_layer_and_collective_counts():
    m = AlphaBetaCommModel()
    sp = m.get_specs("opt-125m")
    assert sp.num_layers == 12
    assert sp.assumed_allreduce_calls_per_token == 24


def test_qwen_layer_and_collective_counts():
    m = AlphaBetaCommModel()
    sp = m.get_specs("qwen2.5-3b")
    assert sp.num_layers == 36
    assert sp.assumed_allreduce_calls_per_token == 72


def test_excess_proxy_matches_hand_calculation_graph_like_numbers():
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
    assert d.observed_excess_us_per_collective_proxy == pytest.approx(
        expected_us, rel=1e-9
    )
    assert d.assumed_num_layers == 12
    assert d.assumed_collectives_per_token == 24
    assert "NOT isolated alpha" in d.proxy_caveat or "not" in d.proxy_caveat.lower()


def test_eager_proxy_much_larger_than_graph_proxy():
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
    assert (
        eager.observed_excess_us_per_collective_proxy
        > 5 * graph.observed_excess_us_per_collective_proxy
    )


def test_tp2_faster_has_no_positive_collective_proxy():
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


def test_invalid_throughput():
    m = AlphaBetaCommModel()
    d = m.diagnose_cell(label="x", model_key="opt-125m", tp1_tok_s=0.0, tp2_tok_s=10.0)
    assert d.measured_regime == "INVALID"


def test_non_tp_comparison_skips_proxy():
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


def test_hypothetical_unknown_payload_does_not_use_concurrency_as_s():
    hyp = HypotheticalAlphaBetaAssumptions(
        alpha_us_per_collective=7.91,
        beta_gb_s=7.8,
        payload_mode="unknown",
        source_note="test",
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
    assert a.hypothetical_t_comm_ms_per_tok == pytest.approx(
        b.hypothetical_t_comm_ms_per_tok
    )


def test_m1_m2_parse_smoke():
    m1 = MilestoneArtifactEvaluator(
        "artifacts/kaggle-2026-09-01-milestone-1"
    ).parse_m1_evidence()
    m2 = MilestoneArtifactEvaluator(
        "artifacts/kaggle-2026-09-02-milestone-2"
    ).parse_m2_concurrency_matrix()
    assert any("opt125m-graph" in c.label for c in m1.cells)
    assert len(m2.cells) == 6
    assert m2.crossover is not None
    assert m2.crossover.measured_throughput_crossover_concurrency == 16
    assert m2.crossover.model_predicted_crossover_concurrency is None
    assert m2.crossover.prediction_status == "unsupported"


def test_report_ascii_and_no_bel(tmp_path: Path):
    paths = generate_reports(
        "artifacts/kaggle-2026-09-01-milestone-1",
        "artifacts/kaggle-2026-09-02-milestone-2",
        tmp_path,
    )
    md = Path(paths["markdown"]).read_text(encoding="utf-8")
    assert "\x07" not in md
    assert "not" in md.lower() and "alpha" in md.lower()
    assert "unsupported" in md.lower()
    # layers wording
    assert "12" in md
    js = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert js["m2"]["crossover"]["prediction_status"] == "unsupported"


def test_cli_main_runs(tmp_path: Path):
    from kaggle_vllm.diagnostics.report import main

    rc = main(
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
