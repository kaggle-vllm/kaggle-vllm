from pathlib import Path

import pytest

from scripts.kaggle_tp_diagnostics import (
    QWEN_SHARDS,
    benchmark_matrix,
    build_parser,
    main,
    validate_qwen_inspection,
)


def test_control_matrix_is_small_and_controlled():
    args = build_parser().parse_args(["--dry-run"])
    matrix = benchmark_matrix(args)
    assert [name for name, _ in matrix] == [
        "opt125m-tp1-eager0",
        "opt125m-tp2-eager0",
        "opt125m-tp1-eager1",
        "opt125m-tp2-eager1",
    ]
    graph = [spec for name, spec in matrix if "eager0" in name]
    assert {spec.tensor_parallel_size for spec in graph} == {1, 2}
    assert all(spec.disable_custom_all_reduce for _, spec in matrix)


def test_qwen_matrix_is_tp2_sharded_and_changes_only_batch_limit(tmp_path):
    args = build_parser().parse_args(
        ["--dry-run", "--qwen-model", str(tmp_path / "existing-model")]
    )
    qwen = [(name, spec) for name, spec in benchmark_matrix(args) if name.startswith("qwen")]
    assert len(qwen) == 2
    baseline = qwen[0][1]
    batched = qwen[1][1]
    assert baseline.tensor_parallel_size == batched.tensor_parallel_size == 2
    assert baseline.model_representation == "sharded_state"
    assert baseline.max_num_batched_tokens is None
    assert batched.max_num_batched_tokens == 4096


def test_dry_run_does_not_create_evidence_directory(tmp_path, capsys):
    output = tmp_path / "evidence"
    assert main(["--dry-run", "--output-dir", str(output)]) == 0
    assert "planned_not_executed" in capsys.readouterr().out
    assert not output.exists()


def test_qwen_inspection_requires_the_validated_four_shard_structure():
    inspection = {
        "rank_count": 2,
        "total_size": sum(QWEN_SHARDS.values()),
        "shards": [
            {"name": name, "size": size} for name, size in QWEN_SHARDS.items()
        ],
    }
    validate_qwen_inspection(inspection)
    inspection["shards"][0]["size"] += 1
    with pytest.raises(RuntimeError, match="structure"):
        validate_qwen_inspection(inspection)


def test_notebook_is_output_free_and_uses_working_directory():
    notebook = Path("kaggle-notebooks/kaggle_vllm_milestone_1_tp_diagnostics.ipynb")
    text = notebook.read_text(encoding="utf-8")
    assert '"execution_count": null' in text
    assert '"outputs": []' in text
    assert "/kaggle/working/kaggle-vllm-tp-milestone-1" in text
