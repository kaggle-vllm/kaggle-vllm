import pytest
from kaggle_vllm.diagnostics.alpha_beta_model import AlphaBetaCommModel
from kaggle_vllm.diagnostics.evaluator import MilestoneArtifactEvaluator


def test_alpha_beta_model_tp1():
    model = AlphaBetaCommModel()
    overhead = model.estimate_comm_overhead_ms("opt-125m", tp_size=1, batch_size=1)
    assert overhead == 0.0


def test_alpha_beta_model_tp2_opt125m():
    model = AlphaBetaCommModel(default_alpha_us=7.91)
    overhead = model.estimate_comm_overhead_ms("opt-125m", tp_size=2, batch_size=1)
    # 24 layers * 2 = 24 calls. Should be > 0
    assert overhead > 0.1


def test_evaluate_cell_crossover():
    model = AlphaBetaCommModel()
    # High concurrency where TP2 > TP1
    estimate = model.evaluate_cell("qwen2.5-3b", tp1_tok_s=138.75, tp2_tok_s=174.27, concurrency=16)
    assert not estimate.is_comm_bound
    assert estimate.observed_delta_percent > 0


def test_evaluator_m1_m2_parsing():
    evaluator_m2 = MilestoneArtifactEvaluator("artifacts/kaggle-2026-09-02-milestone-2")
    m2_results = evaluator_m2.parse_m2_concurrency_matrix()
    assert len(m2_results) == 6
    assert m2_results[0].concurrency == 1
    assert m2_results[3].concurrency == 16
