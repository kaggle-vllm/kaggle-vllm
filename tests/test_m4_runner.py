from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import pytest

from kaggle_vllm.exceptions import BenchmarkError
from scripts import kaggle_m4_multimodel_crossover as runner


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is True
        return [0, *(ord(character) for character in text)]


def arguments(**updates):
    values = {
        "model_key": "example",
        "workload": "short",
        "repetition": 0,
        "mode": "principal",
        "concurrency": None,
        "port": 8000,
        "gpu_memory_utilization": 0.9,
        "telemetry_interval": 0.25,
        "request_timeout": 1800.0,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def model():
    return {
        "hf_id": "example/model",
        "revision": "a" * 40,
        "selected_weight_bytes": 100,
    }


def test_exact_prompt_manifest_is_unique_and_token_controlled():
    manifest = runner.build_prompt_manifest(
        CharacterTokenizer(),
        model_id="example/model",
        revision="a" * 40,
        workload="short",
        repetition=0,
    )
    assert len(manifest["prompts"]) == runner.PROMPT_COUNT
    assert {item["token_count"] for item in manifest["prompts"]} == {128}
    assert len({item["utf8_sha256"] for item in manifest["prompts"]}) == runner.PROMPT_COUNT


def test_principal_and_refinement_matrices_are_exact():
    prompts = ("x", "y")
    matrix = runner.matrix_specs(arguments(), model(), prompts, "0" * 64)
    assert len(matrix) == 12
    assert [spec.tensor_parallel_size for _, spec in matrix] == [1, 2] * 6
    assert [spec.workload.concurrency for _, spec in matrix] == [1, 1, 4, 4, 8, 8, 16, 16, 32, 32, 64, 64]
    assert all(spec.api_mode == "completions" for _, spec in matrix)
    assert all(not spec.enable_prefix_caching for _, spec in matrix)
    assert all(spec.workload.max_output_tokens == 64 for _, spec in matrix)

    refined = runner.matrix_specs(
        arguments(mode="refinement", repetition=5, concurrency=[12, 20]),
        model(),
        prompts,
        "0" * 64,
    )
    assert [spec.workload.concurrency for _, spec in refined] == [12, 12, 20, 20]

    with pytest.raises(Exception, match="requires"):
        runner.matrix_specs(arguments(mode="refinement", repetition=5), model(), prompts, "0" * 64)


def test_prompt_text_is_externalized_from_cell_configuration():
    long_prompt = "evidence " * 200
    spec = runner.matrix_specs(arguments(), model(), (long_prompt,), "0" * 64)[0][1]
    stored = spec.to_dict()["workload"]["prompts"]
    assert stored == [
        {
            "sha256": hashlib.sha256(long_prompt.encode()).hexdigest(),
            "utf8_bytes": len(long_prompt.encode()),
        }
    ]


def test_dry_run_is_side_effect_free(tmp_path, capsys):
    matrix = {
        "models": {
            "example": {
                **model(),
                "architecture": "Example",
                "source_precision": "FP16",
            }
        }
    }
    research = tmp_path / "research"
    research.mkdir()
    (research / "model_matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    assert runner.main([
        "--repository", str(tmp_path),
        "--model-key", "example",
        "--dry-run",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PREPARED_FOR_KAGGLE"
    assert len(payload["cells"]) == 12
    assert not (tmp_path / "m4-evidence").exists()


def test_shard_preparation_creates_absent_root_and_new_child(tmp_path):
    output_root = tmp_path / "nested" / "m4-evidence"

    output = runner.prepare_shard_evidence_directory(output_root, "qwen-shard")

    assert output_root.is_dir()
    assert output == output_root / "qwen-shard"
    assert output.is_dir()


def test_shard_preparation_refuses_existing_child_without_overwrite(tmp_path):
    output_root = tmp_path / "m4-evidence"
    output = runner.prepare_shard_evidence_directory(output_root, "qwen-shard")
    marker = output / "preserved.txt"
    marker.write_text("original evidence", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="refusing to reuse"):
        runner.prepare_shard_evidence_directory(output_root, "qwen-shard")

    assert marker.read_text(encoding="utf-8") == "original evidence"


def test_shard_preparation_rejects_symlink_traversal_before_creation(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)

    with pytest.raises(BenchmarkError, match="traverses a symlink"):
        runner.prepare_shard_evidence_directory(link / "m4-evidence", "qwen-shard")

    assert not (actual / "m4-evidence").exists()


def test_shard_preparation_preserves_kaggle_input_protection():
    with pytest.raises(BenchmarkError, match="/kaggle/input"):
        runner.prepare_shard_evidence_directory(
            Path("/kaggle/input/forbidden-m4-evidence"), "qwen-shard"
        )


def test_m4_source_notebook_execute_cell_preserves_runner_diagnostics():
    notebook_path = Path("kaggle-notebooks/kaggle_vllm_m4_execute_shard.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for cell in code_cells:
        ast.parse("".join(cell["source"]))

    execute = next(cell for cell in code_cells if cell["id"] == "execute")
    source = "".join(execute["source"])
    assert source.index("OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)") < source.index(
        "subprocess.run(command"
    )
    assert "capture_output=True" in source
    assert "completed.stdout" in source
    assert "completed.stderr" in source
    assert "REVIEWED_RETURN_CODES = {0, 2, 3, 4}" in source
    assert "if not output.is_dir():" in source
    assert "raise RuntimeError" in source
    assert "No evidence directory was preserved" not in source
    assert "assert output.is_dir()" not in source
