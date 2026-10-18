"""CPU-only checks for explicit global cp312 selection and import diagnostics."""
from __future__ import annotations

import hashlib
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from kaggle_vllm import KaggleLLM, KaggleSamplingParams
from kaggle_vllm import global_runtime, llm
from kaggle_vllm.exceptions import RuntimeValidationError, VLLMLazyImportError, VLLMNotInstalledError


def test_global_mode_uses_no_managed_stage_and_requires_l4_tp4(monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    checked = []
    monkeypatch.setattr(global_runtime, "validate_global_runtime", lambda: checked.append("versions"))
    monkeypatch.setattr(llm, "validate_l4x4_profile", lambda: checked.append("gpu"))
    monkeypatch.setattr(llm, "activate_runtime", lambda: (_ for _ in ()).throw(AssertionError("managed stage used")))
    monkeypatch.setattr(llm, "detect_kaggle", lambda: False)
    class Fake:
        def __init__(self, **kwargs): self.kwargs = kwargs
    monkeypatch.setattr(llm, "_load_llm_class", lambda **kwargs: Fake)
    wrapped = KaggleLLM(model, profile=llm.L4_PROFILE, runtime_mode="global", strict=False)
    assert checked == ["versions", "gpu"]
    assert wrapped.runtime_mode == "global"
    assert wrapped.engine_args["tensor_parallel_size"] == 4
    assert wrapped.engine_args["model"] == str(model)
    with pytest.raises(RuntimeValidationError, match="tensor_parallel_size=4"):
        KaggleLLM(model, profile=llm.L4_PROFILE, runtime_mode="global", tensor_parallel_size=2)
    with pytest.raises(RuntimeValidationError, match="only the L4x4"):
        KaggleLLM(model, profile=llm.T4_PROFILE, runtime_mode="global")
    with pytest.raises(RuntimeValidationError, match="local_files_only"):
        wrapped.chat([], local_files_only=False)


def test_global_rejects_remote_model_and_disables_hub_network(monkeypatch, tmp_path):
    monkeypatch.setattr(global_runtime, "validate_global_runtime", lambda: {})
    monkeypatch.setattr(llm, "validate_l4x4_profile", lambda: None)
    monkeypatch.setattr(llm, "detect_kaggle", lambda: False)
    monkeypatch.setattr(llm, "_load_llm_class", lambda **kwargs: (_ for _ in ()).throw(AssertionError("imported vLLM")))
    with pytest.raises(RuntimeValidationError, match="local model directory"):
        KaggleLLM("org/remote-model", profile=llm.L4_PROFILE, runtime_mode="global")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    with pytest.raises(AssertionError, match="imported vLLM"):
        KaggleLLM(model, profile=llm.L4_PROFILE, runtime_mode="global")


def test_global_sampling_does_not_activate_managed_runtime(monkeypatch):
    class Params:
        def __init__(self, **kwargs): self.kwargs = kwargs
    monkeypatch.setattr(llm, "activate_runtime", lambda: (_ for _ in ()).throw(AssertionError("managed stage used")))
    monkeypatch.setattr(llm.importlib, "import_module", lambda name: SimpleNamespace(SamplingParams=Params))
    result = KaggleSamplingParams(temperature=0).to_upstream(runtime_mode="global")
    assert result.kwargs == {"temperature": 0}


def test_managed_sampling_keeps_stage_error(monkeypatch):
    monkeypatch.setattr(llm, "activate_runtime", lambda: False)
    with pytest.raises(VLLMNotInstalledError, match="not staged") as caught:
        llm._load_vllm_module(profile=llm.L4_PROFILE)
    assert type(caught.value) is VLLMNotInstalledError


def test_global_wrapper_does_not_open_network_connections(monkeypatch, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(global_runtime, "validate_global_runtime", lambda: {})
    monkeypatch.setattr(llm, "validate_l4x4_profile", lambda: None)
    monkeypatch.setattr(llm, "detect_kaggle", lambda: False)
    class Fake:
        def __init__(self, **kwargs): pass
        def generate(self, prompts, **kwargs): return prompts
    monkeypatch.setattr(llm, "_load_llm_class", lambda **kwargs: Fake)
    monkeypatch.setattr(socket.socket, "connect", lambda *args: (_ for _ in ()).throw(AssertionError("unexpected network access")))
    wrapped = KaggleLLM(model, profile=llm.L4_PROFILE, runtime_mode="global")
    assert wrapped.generate(["offline"]) == ["offline"]


def test_lazy_llm_import_preserves_missing_dependency(monkeypatch):
    failure = ModuleNotFoundError("No module named 'optional_attention'")
    monkeypatch.setattr(llm.importlib, "import_module", lambda name: (_ for _ in ()).throw(failure))
    with pytest.raises(VLLMLazyImportError, match="optional_attention") as caught:
        llm._load_llm_class(allow_t4_fallback=False)
    assert caught.value.__cause__ is failure


def test_t4_fallback_preserves_activation_failure(monkeypatch):
    monkeypatch.setattr(llm.importlib, "import_module", lambda name: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'vllm'", name="vllm")))
    failure = RuntimeError("T4 ABI mismatch")
    monkeypatch.setattr(llm, "activate_t4_runtime", lambda: (_ for _ in ()).throw(failure))
    with pytest.raises(RuntimeError, match="T4 ABI mismatch") as caught:
        llm._load_llm_class()
    assert caught.value is failure


def _fake_global(monkeypatch, tmp_path):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "VLLM_NO_USAGE_STATS"):
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / "dist-packages"
    root.mkdir()
    native = {}
    for name, (relative, _) in global_runtime.NATIVE_EXTENSIONS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"native")
        native[name] = (relative, hashlib.sha256(b"native").hexdigest())
    monkeypatch.setattr(global_runtime, "NATIVE_EXTENSIONS", native)
    monkeypatch.setattr(global_runtime, "_global_site_root", lambda: root)
    monkeypatch.setattr(global_runtime, "sys", SimpleNamespace(
        implementation=SimpleNamespace(name="cpython"), version_info=(3, 12, 0),
        prefix="/usr", base_prefix="/usr", modules={},
    ))
    versions = {
        "kaggle-vllm": "0.3.2", "vllm": "0.30.0", "torch": "2.13.0+cu129",
        "triton": "3.7.1", "torchvision": "0.28.0+cu129", "torchaudio": "2.11.0+cu129",
        "torchcodec": "0.16.0+cu129", "flashinfer-python": "0.6.18.post1",
        "flashinfer-cubin": "0.6.18.post1",
    }
    monkeypatch.setattr(global_runtime, "metadata", SimpleNamespace(distributions=lambda path: [SimpleNamespace(metadata={"Name": n}, version=v, locate_file=lambda _name, root=root: root) for n, v in versions.items()]))
    modules = {name: SimpleNamespace(__file__=str(root / (name.replace(".", "/") + ".so"))) for name in native}
    modules["torch"] = SimpleNamespace(__version__="2.13.0+cu129", version=SimpleNamespace(cuda="12.9"), __file__=str(root / "torch/__init__.py"))
    modules["vllm"] = SimpleNamespace(__version__="0.30.0", __file__=str(root / "vllm/__init__.py"))
    modules["triton"] = SimpleNamespace(__version__="3.7.1", __file__=str(root / "triton/__init__.py"))
    for name in ("kaggle_vllm", "vllm", "torch", "triton"):
        path = root / name / "__init__.py"
        path.parent.mkdir(exist_ok=True)
        path.touch()
    monkeypatch.setattr(global_runtime, "importlib", SimpleNamespace(
        util=SimpleNamespace(find_spec=lambda name: SimpleNamespace(origin=str(root / name / "__init__.py"))),
        import_module=lambda name: modules[name],
    ))
    return root, versions, modules


def test_global_identity_validation(monkeypatch, tmp_path):
    root, versions, modules = _fake_global(monkeypatch, tmp_path)
    report = global_runtime.validate_global_runtime()
    assert report["site_packages"] == str(root)
    assert report["versions"]["vllm"] == "0.30.0"
    assert report["distribution_origins"]["vllm"] == str(root)
    assert report["module_origins"]["vllm"] == str(root / "vllm/__init__.py")
    assert len(report["native_sha256"]) == 2
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "VLLM_NO_USAGE_STATS"):
        assert global_runtime.os.environ[key] == "1"


def test_global_rejects_python_313(monkeypatch, tmp_path):
    root, versions, modules = _fake_global(monkeypatch, tmp_path)
    global_runtime.sys.version_info = (3, 13, 0)
    with pytest.raises(RuntimeValidationError, match="Python 3.13"):
        global_runtime.validate_global_runtime()


@pytest.mark.parametrize("name,wrong", [("vllm", "0.29.0"), ("torch", "2.12.0+cu129"), ("triton", "3.6.0"), ("kaggle-vllm", "0.3.3.dev0")])
def test_global_rejects_wrong_distribution_version(monkeypatch, tmp_path, name, wrong):
    root, versions, modules = _fake_global(monkeypatch, tmp_path)
    versions[name] = wrong
    with pytest.raises(RuntimeValidationError, match=f"{name} must be"):
        global_runtime.validate_global_runtime()


def test_global_rejects_wrong_cuda_and_native_hash(monkeypatch, tmp_path):
    root, versions, modules = _fake_global(monkeypatch, tmp_path)
    modules["torch"].version.cuda = "12.8"
    with pytest.raises(RuntimeValidationError, match="torch/CUDA identity mismatch"):
        global_runtime.validate_global_runtime()
    modules["torch"].version.cuda = "12.9"
    (root / "vllm/_C_stable_libtorch.abi3.so").write_bytes(b"different")
    with pytest.raises(RuntimeValidationError, match="SHA256 mismatch"):
        global_runtime.validate_global_runtime()


def test_global_native_import_error_is_chained(monkeypatch, tmp_path):
    root, versions, modules = _fake_global(monkeypatch, tmp_path)
    failure = ImportError("libtorch ABI mismatch")
    def load(name):
        if name == "vllm._C_stable_libtorch": raise failure
        return modules[name]
    global_runtime.importlib.import_module = load
    with pytest.raises(RuntimeValidationError, match="libtorch ABI mismatch") as caught:
        global_runtime.validate_global_runtime()
    assert caught.value.__cause__ is failure


def test_global_rejects_shadowed_package(monkeypatch, tmp_path):
    root, versions, modules = _fake_global(monkeypatch, tmp_path)
    global_runtime.importlib.util.find_spec = lambda name: SimpleNamespace(origin="/tmp/overlay/vllm/__init__.py" if name == "vllm" else str(root / name / "__init__.py"))
    with pytest.raises(RuntimeValidationError, match="resolves outside"):
        global_runtime.validate_global_runtime()
