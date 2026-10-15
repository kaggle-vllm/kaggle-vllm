"""CPU-only checks of the target-only Kaggle acceptance package."""
from __future__ import annotations

import json
import hashlib
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def asset(name: str) -> dict:
    return runpy.run_path(str(ROOT / "scripts" / name))


def test_acceptance_sources_and_notebook_compile_without_running_gpu():
    for name in ("production_acceptance_v030.py", "production_target_probe_v030.py", "fresh_kaggle_fa2_model_v030.py", "fresh_kaggle_gemma_v030.py", "prepare_kaggle_native_wheel_dataset.py"):
        path = ROOT / "scripts" / name
        compile(path.read_text(), str(path), "exec")
    notebook = json.loads((ROOT / "kaggle-notebooks" / "kaggle_vllm_0_3_0_l4x4_final_acceptance.ipynb").read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
            compile("".join(cell["source"]), "acceptance-notebook-cell", "exec")


def test_competition_runner_has_no_unrelated_model_gate():
    source = (ROOT / "scripts/production_acceptance_v030.py").read_text()
    notebook = (ROOT / "kaggle-notebooks/kaggle_vllm_0_3_0_l4x4_final_acceptance.ipynb").read_text()
    assert "--fa2-model" not in source + notebook
    assert "fresh_kaggle_fa2_model_v030.py" not in source + notebook
    assert "SPLIT_V2_KAGGLE_INPUTS" in source


def test_artifact_discovery_rejects_ambiguity(tmp_path):
    choose = asset("production_acceptance_v030.py")["unique_file"]
    for folder in ("first", "second"):
        p = tmp_path / folder / "runtime.tar.gz"
        p.parent.mkdir()
        p.touch()
    with pytest.raises(RuntimeError, match="exactly one"):
        choose(tmp_path, "runtime.tar.gz")


def test_child_streams_and_cleans_process_group(tmp_path):
    runner_type = asset("production_acceptance_v030.py")["Runner"]
    runner = runner_type(SimpleNamespace(output_root=tmp_path))
    runner.logs.mkdir()
    output = runner.child("fixture", [sys.executable, "-c", "print('streamed output', flush=True)"], timeout=10)
    assert "streamed output" in output
    assert "streamed output" in (runner.logs / "fixture.log").read_text()


def test_child_timeout_kills_its_group(tmp_path):
    runner_type = asset("production_acceptance_v030.py")["Runner"]
    runner = runner_type(SimpleNamespace(output_root=tmp_path))
    runner.logs.mkdir()
    with pytest.raises(TimeoutError):
        runner.child("timeout", [sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)


def test_sdk_install_uses_exact_offline_wheel_even_if_version_is_present(tmp_path):
    runner_type = asset("production_acceptance_v030.py")["Runner"]
    runner = runner_type(SimpleNamespace(output_root=tmp_path))
    runner.sdk = tmp_path / "kaggle_vllm-0.3.0-py3-none-any.whl"
    runner.artifact_hash = lambda *args: {"sha256": "fixture"}
    calls = []
    runner.child = lambda name, command, **options: calls.append((name, command, options))
    runner.install()
    name, command, options = calls[0]
    assert name == "06_sdk_install"
    assert command[:4] == [sys.executable, "-m", "pip", "install"]
    assert {"--no-index", "--no-deps", "--no-cache-dir", "--force-reinstall"} <= set(command)
    assert command[-1] == str(runner.sdk)
    assert options["timeout"] == 300


def test_competition_matrix_never_claims_real_model_fa2(tmp_path):
    runner_type = asset("production_acceptance_v030.py")["Runner"]
    runner = runner_type(SimpleNamespace(output_root=tmp_path, runtime_mode="SPLIT_V2_KAGGLE_INPUTS"))
    runner.evidence.mkdir()
    runner.records = [
        {"step": 0, "name": "required fixture", "required_for_core": True, "status": "PASS"},
        {"step": 18, "name": "cleanup fixture", "required_for_core": True, "status": "PASS"},
    ]
    result = runner.matrix()
    saved = json.loads((runner.evidence / "19_final_matrix.json").read_text())
    assert result == {"core_release_pass": True, "fa2_model_runtime_pass": False}
    assert saved["status"] == "FRESH_KAGGLE_PASS"
    assert saved["FA2_MODEL_RUNTIME_PASS"] is False
    assert saved["fa2_model_status"] == "NOT_PROVEN_OUTSIDE_COMPETITION"


def test_final_sdk_notebook_uses_only_verified_candidate_runner():
    path = ROOT / "kaggle-notebooks/kaggle_vllm_0_3_0_l4x4_final_sdk_acceptance.ipynb"
    notebook = json.loads(path.read_text())
    assert len(notebook["cells"]) == 4
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for cell in notebook["cells"]:
        assert cell["cell_type"] == "code"
        compile("".join(cell["source"]), str(path), "exec")
    assert "KAGGLE_VLLM_INPUT_ROOT" in source
    assert 'CANDIDATE_DATASET_ROOT / "production_acceptance_v030.py"' in source
    assert '"--input-root", str(INPUT_ROOT), "--sdk-wheel", str(SDK_WHEEL)' in source
    assert '"--native-wheel"' in source
    assert '"--cleanup-runtime-after-pass"' in source
    assert 'INPUT_ROOT.rglob(' not in source
    assert source.count('19_final_matrix.json') == 1


def test_matrix_is_written_once_and_cleanup_requires_verified_pass(tmp_path, monkeypatch):
    runner_type = asset("production_acceptance_v030.py")["Runner"]
    runner = runner_type(SimpleNamespace(output_root=tmp_path, runtime_mode="SPLIT_V2_KAGGLE_INPUTS"))
    runner.evidence.mkdir()
    runner.records = [{"step": 0, "name": "fixture", "required_for_core": True, "status": "PASS"}]
    writes = []
    original_write = Path.write_text
    def counted_write(path, *args, **kwargs):
        if path.name == "19_final_matrix.json":
            writes.append(path)
        return original_write(path, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", counted_write)
    runner.matrix()
    assert len(writes) == 1
    archive = tmp_path / "20_evidence_bundle.zip"
    archive.write_bytes(b"verified evidence")
    runner.records.append({"step": 20, "status": "PASS", "evidence": {"sha256": "wrong"}})
    with pytest.raises(RuntimeError, match="SHA-verified"):
        runner.cleanup_runtime_after_pass({"status": "FRESH_KAGGLE_PASS"})
    with pytest.raises(RuntimeError, match="FRESH_KAGGLE_PASS"):
        runner.cleanup_runtime_after_pass({"status": "FRESH_KAGGLE_FAIL"})
    assert not (tmp_path / "post_pass_runtime_cleanup.json").exists()


def test_post_pass_cleanup_uses_sdk_owned_reset_only_after_verified_bundle(tmp_path, monkeypatch):
    from kaggle_vllm import l4_bootstrap
    runner_type = asset("production_acceptance_v030.py")["Runner"]
    runner = runner_type(SimpleNamespace(output_root=tmp_path))
    archive = tmp_path / "20_evidence_bundle.zip"
    archive.write_bytes(b"complete evidence")
    runner.records = [{"step": 20, "status": "PASS", "evidence": {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}}]
    calls = []
    monkeypatch.setattr(l4_bootstrap, "reset_runtime", lambda target, *, yes: calls.append((target, yes)) or {"action": "remove"})
    assert runner.cleanup_runtime_after_pass({"status": "FRESH_KAGGLE_PASS"}) == {"action": "remove"}
    assert calls == [(asset("production_acceptance_v030.py")["TARGET"], True)]
    assert json.loads((tmp_path / "post_pass_runtime_cleanup.json").read_text()) == {"action": "remove"}


def test_native_dataset_helper_rejects_wrong_hash_without_output(tmp_path, monkeypatch):
    helper = asset("prepare_kaggle_native_wheel_dataset.py")
    archive = tmp_path / "p31-kaggle-vllm-v030-l4x4-final-v2.tar.gz"
    archive.write_bytes(b"wrong")
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["helper", "--p31-archive", str(archive), "--output", str(output)])
    with pytest.raises(SystemExit, match="mismatch"):
        helper["main"]()
    assert not output.exists()
