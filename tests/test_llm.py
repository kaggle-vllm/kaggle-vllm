from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggle_vllm.exceptions import ShardedModelError, VLLMNotInstalledError
from kaggle_vllm.llm import KaggleLLM, _load_llm_class


class FakeUpstreamLLM:
    last_kwargs = None

    def __init__(self, **kwargs):
        FakeUpstreamLLM.last_kwargs = kwargs
        self.llm_engine = SimpleNamespace(
            engine_core=SimpleNamespace(save_sharded_state=self._save)
        )

    def generate(self, *args, **kwargs):
        return args, kwargs

    @staticmethod
    def _save(path, max_size=None):
        root = Path(path)
        (root / "model-rank-0-part-0.safetensors").write_bytes(b"0")
        (root / "model-rank-1-part-0.safetensors").write_bytes(b"1")
        (root / "config.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def make_llm(monkeypatch, **kwargs):
    monkeypatch.setattr(
        "kaggle_vllm.llm.validate_tensor_parallel_size", lambda size: None
    )
    monkeypatch.setattr("kaggle_vllm.llm._load_llm_class", lambda: FakeUpstreamLLM)
    return KaggleLLM("model-id", **kwargs)


def test_import_diagnostics_does_not_import_vllm():
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, kaggle_vllm.environment; print('vllm' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.strip() == "False"


def test_plain_import_has_no_network_or_install_side_effects():
    root = Path(__file__).parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    code = """
import sys
events = []
def audit(event, args):
    if event.startswith('socket.') or event == 'subprocess.Popen':
        events.append(event)
sys.addaudithook(audit)
import kaggle_vllm
print(events)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.strip() == "[]"


def test_missing_vllm_raises_useful_error(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr("kaggle_vllm.llm.importlib.import_module", missing)
    with pytest.raises(VLLMNotInstalledError, match="bootstrap"):
        _load_llm_class()


def test_t4_conservative_defaults_and_kwargs_forwarding(monkeypatch):
    make_llm(monkeypatch, tensor_parallel_size=2, max_model_len=2048)
    assert FakeUpstreamLLM.last_kwargs == {
        "dtype": "float16",
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "max_model_len": 2048,
        "model": "model-id",
        "tensor_parallel_size": 2,
    }


def test_user_overrides_defaults_and_sharded_load_format(monkeypatch):
    make_llm(
        monkeypatch,
        dtype="float32",
        enforce_eager=False,
        disable_custom_all_reduce=False,
        load_format="sharded_state",
    )
    assert FakeUpstreamLLM.last_kwargs["dtype"] == "float32"
    assert FakeUpstreamLLM.last_kwargs["enforce_eager"] is False
    assert FakeUpstreamLLM.last_kwargs["disable_custom_all_reduce"] is False
    assert FakeUpstreamLLM.last_kwargs["load_format"] == "sharded_state"


def test_generate_and_save_delegate_to_upstream(monkeypatch, tmp_path):
    llm = make_llm(monkeypatch, tensor_parallel_size=2)
    assert llm.generate(["hello"], temperature=0) == ((["hello"],), {"temperature": 0})
    inspection = llm.save_sharded_model(tmp_path / "sharded", max_size=123)
    assert inspection.rank_count == 2
    assert inspection.valid


def test_save_rejects_symlinked_destination(monkeypatch, tmp_path):
    llm = make_llm(monkeypatch, tensor_parallel_size=2)
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ShardedModelError, match="traverses a symlink"):
        llm.save_sharded_model(link)
