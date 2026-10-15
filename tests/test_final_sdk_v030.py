"""CPU-only checks for the additive final SDK candidate boundaries."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggle_vllm import KaggleLLM, discover_gemma4_qat, load_profile
from kaggle_vllm.cli_v3 import build_parser, main
from kaggle_vllm.exceptions import BootstrapError, RuntimeValidationError
from kaggle_vllm.l4_bootstrap import check_free_space, reset_runtime
from kaggle_vllm.profile_v3 import WHEEL_SHA256
from kaggle_vllm.public_artifacts_v3 import immutable_revision
from kaggle_vllm.runtime_v3 import lean_runtime_manifest, pinned_runtime_manifest


def test_p33_lock_is_distinct_from_archival_closure():
    lean = lean_runtime_manifest()
    archival = pinned_runtime_manifest()
    assert len(lean["wheels"]) == 168
    assert len(archival["wheels"]) == 195
    assert lean["native"]["sha256"] == WHEEL_SHA256
    assert len({row["filename"] for row in lean["wheels"]}) == 168
    assert {"pyproject_hooks", "semantic_version", "vcs_versioning"} <= {row["filename"].split("-")[0] for row in lean["wheels"]}


def test_historical_t4_profile_remains_loadable():
    historical = load_profile("kaggle-t4x2-cu128")
    assert historical.gpu_count == 2
    assert historical.compute_capability == (7, 5)
    assert historical.vllm_source_tag == "v0.18.1"
    assert historical.hf_revision == "f6b4f10de54924ed6fe9e28cceab84eca7276ab6"


def test_l4_cli_requires_explicit_profile():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["bootstrap", "--offline"])
    args = build_parser().parse_args(["bootstrap", "--profile", "kaggle-l4x4-cu129", "--offline"])
    assert args.runtime_root == Path("/tmp/kaggle-vllm/v0.3.0/l4x4")
    assert build_parser().parse_args(["plan", "--profile", "kaggle-l4x4-cu129"]).command == "plan"


def test_discover_model_cli_lists_local_config(tmp_path, capsys):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    assert main(["discover-model", "--root", str(tmp_path)]) == 0
    assert str(model) in capsys.readouterr().out


def test_evidence_export_is_atomic_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr("kaggle_vllm.doctor_v3.doctor_report", lambda **kwargs: {"compatible": True, "findings": []})
    output = tmp_path / "doctor.json"
    args = ["evidence-export", "--profile", "kaggle-l4x4-cu129", "--output", str(output)]
    assert main(args) == 0
    assert json.loads(output.read_text())["compatible"] is True
    with pytest.raises(SystemExit):
        main(args)


def test_unpublished_public_revision_fails_closed():
    with pytest.raises(BootstrapError, match="immutable"):
        immutable_revision()


def test_free_space_guard_reports_required_and_available(monkeypatch):
    monkeypatch.setattr("kaggle_vllm.l4_bootstrap.shutil.disk_usage", lambda _: SimpleNamespace(free=1024))
    with pytest.raises(BootstrapError, match="available 1024 bytes, required 2048 bytes"):
        check_free_space("/tmp/kaggle-vllm/v0.3.0/l4x4", required=2048)


def test_reset_requires_manifest_and_yes(tmp_path):
    target = tmp_path / "kaggle-vllm-v030-runtime"
    target.mkdir()
    with pytest.raises(BootstrapError, match="ownership manifest"):
        reset_runtime(target, yes=True)
    manifest = target / "manifests" / "sdk-runtime.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"profile": "kaggle-l4x4-cu129", "native_sha256": WHEEL_SHA256}))
    assert reset_runtime(target, dry_run=True)["action"] == "remove"
    assert target.exists()
    with pytest.raises(BootstrapError, match="requires --yes"):
        reset_runtime(target)
    reset_runtime(target, yes=True)
    assert not target.exists()


def test_failed_offline_pip_rolls_back_partial_target(tmp_path, monkeypatch):
    from kaggle_vllm.l4_bootstrap import bootstrap
    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "kaggle-vllm-v030-runtime"
    monkeypatch.setattr("kaggle_vllm.l4_bootstrap.discover_runtime", lambda *a, **kw: source)
    monkeypatch.setattr("kaggle_vllm.l4_bootstrap._compatibility", lambda strict: [])
    monkeypatch.setattr("kaggle_vllm.l4_bootstrap.check_free_space", lambda *a, **kw: {})
    monkeypatch.setattr("kaggle_vllm.l4_bootstrap.verify_runtime", lambda *a, **kw: {"wheels": []})
    monkeypatch.setattr("kaggle_vllm.l4_bootstrap.lean_runtime_manifest", lambda: {"wheels": []})
    def failed_pip(*args, **kwargs):
        staging = target / next(p.name for p in target.iterdir() if p.name.startswith(".pydeps-"))
        (staging / "partial.txt").write_text("partial")
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr("kaggle_vllm.l4_bootstrap.subprocess.run", failed_pip)
    with pytest.raises(BootstrapError, match="offline wheel staging failed"):
        bootstrap(runtime_root=source, target=target)
    assert not (target / "pydeps").exists()
    assert not any(target.glob(".pydeps-*"))


def test_gemma_discovery_chooses_latest_numbered_version(tmp_path):
    for version in ("1", "2", "10"):
        root = tmp_path / version
        root.mkdir()
        (root / "config.json").write_text("{}")
    assert discover_gemma4_qat(tmp_path).name == "10"
    (tmp_path / "010").mkdir()
    (tmp_path / "010" / "config.json").write_text("{}")
    with pytest.raises(RuntimeValidationError, match="ambiguous"):
        discover_gemma4_qat(tmp_path)


def test_explicit_t4_wrapper_keeps_tp2(monkeypatch):
    class Fake:
        def __init__(self, **kwargs): self.kwargs = kwargs
    monkeypatch.setattr("kaggle_vllm.llm.detect_kaggle", lambda: False)
    monkeypatch.setattr("kaggle_vllm.llm._load_llm_class", lambda: Fake)
    wrapped = KaggleLLM("historical-model", profile="kaggle-t4x2-cu128")
    assert wrapped.engine_args["tensor_parallel_size"] == 2
    assert wrapped.upstream.kwargs["model"] == "historical-model"


def test_implicit_local_wrapper_keeps_v020_t4_defaults(monkeypatch):
    class Fake:
        def __init__(self, **kwargs): self.kwargs = kwargs
    monkeypatch.setattr("kaggle_vllm.llm.detect_kaggle", lambda: False)
    monkeypatch.setattr("kaggle_vllm.llm._load_llm_class", lambda: Fake)
    wrapped = KaggleLLM("historical-model")
    assert wrapped.profile == "kaggle-t4x2-cu128"
    assert wrapped.engine_args["tensor_parallel_size"] == 1
    assert wrapped.engine_args["dtype"] == "float16"


def test_t4_rejects_active_l4_overlay(monkeypatch):
    monkeypatch.setattr("kaggle_vllm.llm.detect_kaggle", lambda: False)
    monkeypatch.setattr("kaggle_vllm.l4_bootstrap._ACTIVE_L4_TARGET", Path("/tmp/kaggle-vllm/v0.3.0/l4x4"))
    with pytest.raises(RuntimeValidationError, match="L4 runtime is active"):
        KaggleLLM("historical-model", profile="kaggle-t4x2-cu128")


def test_auto_l4_detection_does_not_preload_base_torch(monkeypatch):
    class Fake:
        def __init__(self, **kwargs): self.kwargs = kwargs
    monkeypatch.setattr("kaggle_vllm.llm.detect_kaggle", lambda: True)
    monkeypatch.setattr("kaggle_vllm.llm.collect", lambda: (_ for _ in ()).throw(AssertionError("base Torch must not load")))
    monkeypatch.setattr("kaggle_vllm.llm.subprocess.run", lambda *args, **kwargs: SimpleNamespace(stdout="NVIDIA L4, 8.9\n" * 4))
    monkeypatch.setattr("kaggle_vllm.llm.activate_runtime", lambda: True)
    monkeypatch.setattr("kaggle_vllm.llm.validate_l4x4_profile", lambda: None)
    monkeypatch.setattr("kaggle_vllm.llm._load_llm_class", lambda: Fake)
    wrapped = KaggleLLM("l4-model")
    assert wrapped.profile == "kaggle-l4x4-cu129"
    assert wrapped.engine_args["tensor_parallel_size"] == 4


def test_strict_bootstrap_inventory_does_not_import_torch(monkeypatch):
    from kaggle_vllm.l4_bootstrap import _compatibility
    monkeypatch.setattr("kaggle_vllm.environment.detect_kaggle", lambda: True)
    monkeypatch.setattr("kaggle_vllm.l4_bootstrap.subprocess.run", lambda *args, **kwargs: SimpleNamespace(stdout="NVIDIA L4, 8.9\n" * 4))
    problems = _compatibility(True)
    assert not any("GPU" in item for item in problems)
    assert any("cp312" in item for item in problems) == (sys.version_info[:2] != (3, 12))
