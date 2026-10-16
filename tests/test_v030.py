"""CPU-only release checks; no native cp312 wheel is imported or executed."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import kaggle_vllm
from kaggle_vllm import KaggleLLM, KaggleSamplingParams
from kaggle_vllm.cuda_driver import ensure_cuda_driver_linker_alias
from kaggle_vllm.environment import GPUInfo
from kaggle_vllm.exceptions import BootstrapError, ProfileError, RuntimeValidationError
from kaggle_vllm.profile_v3 import WHEEL_FILENAME, WHEEL_SHA256, load_profile, validate_profile
from kaggle_vllm.runtime import all_nvidia_l4, all_sm89, validate_l4x4_profile, validate_tensor_parallel_size
from kaggle_vllm.runtime_v3 import BUNDLE_NAME, discover_runtime, load_runtime_manifest, sha256, validate_native_wheel


def gpus(name="NVIDIA L4", cap=(8, 9), count=4):
    return tuple(GPUInfo(i, name, cap) for i in range(count))


def test_version():
    assert kaggle_vllm.__version__ == "0.3.2"


def test_root_cli_help_lists_new_and_historical_commands(capsys):
    from kaggle_vllm.cli_v3 import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for command in ("profile", "plan", "verify-hashes", "discover-model", "evidence-export", "serve", "benchmark"):
        assert command in help_text


def test_profile_identity():
    p = load_profile()
    assert (p.name, p.python_abi, p.gpu_count, p.compute_capability) == ("kaggle-l4x4-cu129", "cp312", 4, (8, 9))
    assert p.wheel_sha256 == WHEEL_SHA256


def test_profile_malformed():
    d = copy.deepcopy(load_profile().data)
    d["gpu"]["name"] = "NVIDIA Tesla T4"
    with pytest.raises(ProfileError): validate_profile(d)


def test_unsupported_profile():
    with pytest.raises(ProfileError): load_profile("kaggle-t4x2-cu128")


def test_l4_exact_name():
    assert all_nvidia_l4(gpus())
    assert not all_nvidia_l4(gpus("NVIDIA L4 virtual"))
    assert not all_nvidia_l4(gpus("NVIDIA Tesla T4"))


def test_sm89():
    assert all_sm89(gpus())
    assert not all_sm89(gpus(cap=(8, 0)))


@pytest.mark.parametrize("name,cap,count", [("NVIDIA L4", (8, 9), 1), ("NVIDIA L4", (8, 9), 2), ("NVIDIA Tesla T4", (7, 5), 4), ("NVIDIA A100", (8, 0), 4), ("NVIDIA P100", (6, 0), 4), ("NVIDIA L4", (8, 0), 4)])
def test_reject_wrong_gpu(name, cap, count):
    with pytest.raises(RuntimeValidationError): validate_l4x4_profile(SimpleNamespace(gpu_count=count, gpus=gpus(name, cap, count)))


def test_accept_exact_gpu():
    validate_l4x4_profile(SimpleNamespace(gpu_count=4, gpus=gpus()))


def test_default_tp_and_kwargs(monkeypatch):
    class Fake:
        def __init__(self, **kwargs): self.kwargs = kwargs
        def generate(self, *args, **kwargs): return (args, kwargs)
    monkeypatch.setattr("kaggle_vllm.llm._load_llm_class", lambda: Fake)
    monkeypatch.setattr("kaggle_vllm.llm.activate_runtime", lambda: True)
    monkeypatch.setattr("kaggle_vllm.llm.detect_kaggle", lambda: False)
    llm = KaggleLLM("model", profile="kaggle-l4x4-cu129", max_model_len=2048)
    assert llm.engine_args == {"enforce_eager": True, "disable_custom_all_reduce": True, "trust_remote_code": False, "max_model_len": 2048, "model": "model", "tensor_parallel_size": 4}
    assert llm.upstream.kwargs == llm.engine_args
    assert "VLLM_ATTENTION_BACKEND" not in os.environ or llm.engine_args.get("attention_backend") != "FLASH_ATTN"


def test_tp_gt_visible_rejected():
    with pytest.raises(RuntimeValidationError): validate_tensor_parallel_size(5, gpu_count=4)


def test_lazy_import():
    code = "import sys,kaggle_vllm;print('vllm' in sys.modules,'torch' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert result.stdout.strip() == "False False"


def test_sampling_lazy(monkeypatch):
    params = KaggleSamplingParams(temperature=0, max_tokens=64)
    class FakeParams:
        def __init__(self, **kwargs): self.kwargs = kwargs
    monkeypatch.setattr("kaggle_vllm.llm._load_vllm_module", lambda: SimpleNamespace(SamplingParams=FakeParams))
    assert params.to_upstream().kwargs == {"temperature": 0, "max_tokens": 64}


def test_sha256(tmp_path):
    p = tmp_path / "x"
    p.write_bytes(b"abc")
    assert sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_native_filename_and_hash(tmp_path):
    p = tmp_path / WHEEL_FILENAME
    p.write_bytes(b"wrong")
    with pytest.raises(BootstrapError): validate_native_wheel(p)
    with pytest.raises(BootstrapError): validate_native_wheel(tmp_path / "vllm-0.30.0-cp311-cp311-linux_x86_64.whl")


def test_native_abi_matches_interpreter():
    from kaggle_vllm.l4_bootstrap import _compatibility
    problems = _compatibility(False)
    assert any("cp312" in x for x in problems) == (sys.version_info[:2] != (3, 12))


def test_offline_discovery(tmp_path):
    p = tmp_path / "nested" / (BUNDLE_NAME + ".tar.gz")
    p.parent.mkdir()
    p.touch()
    assert discover_runtime(tmp_path) == p


def test_ambiguous_discovery(tmp_path):
    for n in ("a", "b"):
        p = tmp_path / n / (BUNDLE_NAME + ".tar.gz")
        p.parent.mkdir()
        p.touch()
    with pytest.raises(BootstrapError): discover_runtime(tmp_path)


def test_manifest_rejects_old_torchcodec(tmp_path):
    root = tmp_path
    p = root / "metadata" / "RUNTIME_MANIFEST.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"schema_version": 3, "profile": "kaggle-l4x4-cu129", "native": {"filename": WHEEL_FILENAME, "sha256": WHEEL_SHA256}, "wheel_count": 1, "wheels": [{"filename": "torchcodec-0.17.0.whl", "distribution": "torchcodec", "version": "0.17.0", "size": 1, "sha256": "0" * 64}]}))
    from kaggle_vllm import runtime_v3
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runtime_v3, "validate_native_wheel", lambda x: x)
    monkeypatch.setattr(runtime_v3, "MANIFEST_SHA256", sha256(p))
    try:
        with pytest.raises(BootstrapError): runtime_v3.verify_runtime(root, full=False)
    finally: monkeypatch.undo()


def test_cuda_driver_path_safety(tmp_path):
    real = tmp_path / "libcuda.so.1"
    real.touch()
    with pytest.raises(RuntimeValidationError): ensure_cuda_driver_linker_alias(tmp_path / "outside", driver=real)


def test_cuda_driver_versioned_target_symlink(tmp_path, monkeypatch):
    from kaggle_vllm import cuda_driver
    monkeypatch.setattr(cuda_driver, "SDK_ROOT", tmp_path / "sdk")
    driver_dir = tmp_path / "driver"
    driver_dir.mkdir()
    versioned = driver_dir / "libcuda.so.580.1"
    versioned.touch()
    driver = driver_dir / "libcuda.so.1"
    driver.symlink_to(versioned)
    alias_dir = tmp_path / "sdk" / "cuda-driver-link"
    result = cuda_driver.ensure_cuda_driver_linker_alias(alias_dir, driver=driver)
    assert result["status"] == "PASS"
    assert (alias_dir / "libcuda.so").readlink() == driver
    assert (alias_dir / "libcuda.so").resolve() == versioned


def test_bootstrap_dry_run_no_mutation(tmp_path, monkeypatch):
    from kaggle_vllm.l4_bootstrap import bootstrap
    root = tmp_path / BUNDLE_NAME
    p = root / "metadata" / "RUNTIME_MANIFEST.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"schema_version": 3, "profile": "kaggle-l4x4-cu129", "native": {"filename": WHEEL_FILENAME, "sha256": WHEEL_SHA256}, "wheel_count": 1, "wheels": [{"filename": "torchcodec-0.16.0+cu129.whl", "distribution": "torchcodec", "version": "0.16.0+cu129", "size": 1, "sha256": "0" * 64}]}))
    (root / "native").mkdir()
    (root / "native" / WHEEL_FILENAME).touch()
    monkeypatch.setattr("kaggle_vllm.runtime_v3.MANIFEST_SHA256", sha256(p))
    target = tmp_path / "target"
    result = bootstrap(runtime_root=root, target=target, dry_run=True)
    assert result["dry_run"] and not target.exists()


def test_doctor_json_schema(monkeypatch):
    from kaggle_vllm import doctor_v3
    env = SimpleNamespace(is_kaggle=False, torch=None, torch_cuda=None, cuda_available=False, gpu_count=0, gpus=(), gpu_names=[], gpu_caps=[], cuda_driver=None)
    monkeypatch.setattr(doctor_v3, "collect", lambda: env)
    report = doctor_v3.doctor_report()
    assert report["schema_version"] == 3 and report["compatible"] is False
    assert isinstance(report["findings"], list)


def test_dependency_reachability_extras_and_build_exclusion():
    from kaggle_vllm.dependency_graph_v3 import reachable_distributions
    reqs = {
        "vllm": ["nvidia-cutlass-dsl==4.7.1", "fastapi[standard]"],
        "nvidia-cutlass-dsl": ["nvidia-cutlass-dsl-libs-base==4.7.1"],
        "nvidia-cutlass-dsl-libs-base": ["nvidia-cuda-nvdisasm<14,>=13.3"],
        "nvidia-cuda-nvdisasm": [],
        "fastapi": ["uvicorn; extra == 'standard'"],
        "uvicorn": [],
        "cmake": [],
    }
    closure, edges = reachable_distributions(reqs, ["vllm"])
    assert "nvidia-cuda-nvdisasm" in closure
    assert "uvicorn" in closure
    assert "cmake" not in closure
    assert any(x[1] == "nvidia-cuda-nvdisasm" for x in edges)


def test_generate_converts_sampling_params_lazily(monkeypatch):
    class Fake:
        def __init__(self, **kwargs): pass
        def generate(self, *args, **kwargs): return args, kwargs
    monkeypatch.setattr("kaggle_vllm.llm._load_llm_class", lambda: Fake)
    monkeypatch.setattr("kaggle_vllm.llm.detect_kaggle", lambda: False)
    monkeypatch.setattr(KaggleSamplingParams, "to_upstream", lambda self, profile=None: "upstream-params")
    llm = KaggleLLM("model")
    args, kwargs = llm.generate(["prompt"], KaggleSamplingParams(max_tokens=3), use_tqdm=False)
    assert args == (["prompt"], "upstream-params")
    assert kwargs == {"use_tqdm": False}


def test_strict_profile_requires_tp4(monkeypatch):
    monkeypatch.setattr("kaggle_vllm.llm.detect_kaggle", lambda: True)
    monkeypatch.setattr("kaggle_vllm.llm.activate_runtime", lambda: True)
    monkeypatch.setattr("kaggle_vllm.llm.validate_l4x4_profile", lambda: None)
    with pytest.raises(RuntimeValidationError, match="tensor_parallel_size=4"):
        KaggleLLM("model", tensor_parallel_size=2, profile="kaggle-l4x4-cu129")


def test_chat_uses_processor_template(monkeypatch):
    class Fake:
        def __init__(self, **kwargs): pass
        def generate(self, prompts, *args, **kwargs): return prompts
    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs == {"local_files_only": True, "trust_remote_code": False}
            return cls()
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "rendered prompt"
    monkeypatch.setattr("kaggle_vllm.llm._load_llm_class", lambda: Fake)
    monkeypatch.setattr("kaggle_vllm.llm.detect_kaggle", lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=Processor, AutoTokenizer=Processor))
    assert KaggleLLM("model").chat([{"role": "user", "content": "hello"}]) == ["rendered prompt"]


def test_split_v2_discovery_prefers_corrected_components(tmp_path):
    from kaggle_vllm.runtime_v3 import SPLIT_GROUPS, REQUIRED_DATASET_GROUPS, PURE_WHEEL_DIR, SplitRuntimeSource
    for group in REQUIRED_DATASET_GROUPS:
        directory = tmp_path / f"kaggle-vllm-v030-py312-{group}-v2"
        directory.mkdir()
        (directory / ("WHEELHOUSE_MANIFEST.json" if group == "wheelhouse-audit" else f"{group}.whl")).touch()
    pure = tmp_path / PURE_WHEEL_DIR
    pure.mkdir()
    native = tmp_path / WHEEL_FILENAME
    native.touch()
    found = discover_runtime(tmp_path)
    assert isinstance(found, SplitRuntimeSource)
    assert found.native == native and set(found.components) == set(SPLIT_GROUPS)
    assert found.components["build-deps"] == pure
    assert found.audit == tmp_path / "kaggle-vllm-v030-py312-wheelhouse-audit-v2"


def test_split_duplicate_corrected_dataset_rejected(tmp_path):
    from kaggle_vllm.runtime_v3 import discover_split_components
    for parent in ("first", "second"):
        directory = tmp_path / parent / "kaggle-vllm-v030-py312-runtime-core-v2"
        directory.mkdir(parents=True)
        (directory / "fixture.whl").touch()
    with pytest.raises(BootstrapError, match="found 2"):
        discover_split_components(tmp_path)


def test_split_extracted_and_zip_views_reject_other_mount_duplicate(tmp_path):
    from kaggle_vllm.runtime_v3 import _component
    slug = "kaggle-vllm-v030-py312-runtime-core-v2"
    extracted = tmp_path / "first" / slug
    extracted.mkdir(parents=True)
    (extracted / "fixture.whl").touch()
    # A source ZIP in the same Dataset mount is another view of that Dataset.
    colocated = extracted / (slug + ".zip")
    colocated.touch()
    assert _component(tmp_path, "runtime-core") == extracted
    sibling = extracted.parent / (slug + ".zip")
    sibling.touch()
    with pytest.raises(BootstrapError, match="ambiguous"):
        _component(tmp_path, "runtime-core")
    sibling.unlink()
    duplicate = tmp_path / "second" / (slug + ".zip")
    duplicate.parent.mkdir()
    duplicate.touch()
    with pytest.raises(BootstrapError, match="ambiguous"):
        _component(tmp_path, "runtime-core")


def test_split_zip_only_dataset_views_use_small_pure_wheel_fallback(tmp_path):
    from kaggle_vllm.runtime_v3 import REQUIRED_DATASET_GROUPS, PURE_WHEEL_DIR, discover_split_components
    for group in REQUIRED_DATASET_GROUPS:
        dataset = tmp_path / f"kaggle-vllm-v030-py312-{group}-v2"
        dataset.mkdir()
        (dataset / (dataset.name + ".zip")).touch()
    pure = tmp_path / "kaggle-vllm-v030-production-acceptance" / PURE_WHEEL_DIR
    pure.mkdir(parents=True)
    components, audit = discover_split_components(tmp_path)
    assert components["build-deps"] == pure
    for group in REQUIRED_DATASET_GROUPS:
        selected = audit if group == "wheelhouse-audit" else components[group]
        assert selected.name == f"kaggle-vllm-v030-py312-{group}-v2.zip"


def test_split_explicit_dataset_directory_resolves_its_zip(tmp_path):
    from kaggle_vllm.runtime_v3 import REQUIRED_DATASET_GROUPS, PURE_WHEEL_DIR, discover_split_components
    overrides = {}
    for group in REQUIRED_DATASET_GROUPS:
        dataset = tmp_path / f"kaggle-vllm-v030-py312-{group}-v2"
        dataset.mkdir()
        (dataset / (dataset.name + ".zip")).touch()
        overrides[group] = dataset
    pure = tmp_path / PURE_WHEEL_DIR
    pure.mkdir()
    (pure / "packaging-26.3-py3-none-any.whl").touch()
    overrides["build-deps"] = pure
    components, audit = discover_split_components(tmp_path, overrides=overrides)
    assert components["build-deps"] == pure
    for group in REQUIRED_DATASET_GROUPS:
        selected = audit if group == "wheelhouse-audit" else components[group]
        assert selected == overrides[group] / (overrides[group].name + ".zip")


def test_native_discovery_prioritizes_p31_notebook_output_and_rejects_ties(tmp_path):
    from kaggle_vllm.runtime_v3 import P31_ARCHIVE_NAME, discover_native_source
    datasets = tmp_path / "datasets"
    notebooks = tmp_path / "notebooks"
    datasets.mkdir(); notebooks.mkdir()
    dedicated = datasets / WHEEL_FILENAME
    dedicated.touch()
    archive = notebooks / P31_ARCHIVE_NAME
    archive.touch()
    assert discover_native_source(datasets, notebook_root=notebooks) == archive
    wheel = notebooks / WHEEL_FILENAME
    wheel.touch()
    assert discover_native_source(datasets, notebook_root=notebooks) == wheel
    assert discover_native_source(datasets, explicit=dedicated, notebook_root=notebooks) == dedicated
    second = notebooks / "another" / WHEEL_FILENAME
    second.parent.mkdir(); second.touch()
    with pytest.raises(BootstrapError, match="ambiguous"):
        discover_native_source(datasets, notebook_root=notebooks)


def test_p31_tar_native_member_must_be_exact_and_regular(tmp_path, monkeypatch):
    import io
    import tarfile
    from kaggle_vllm import runtime_v3
    archive = tmp_path / runtime_v3.P31_ARCHIVE_NAME
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("wrong-subdirectory/" + WHEEL_FILENAME)
        member.size = 3
        tar.addfile(member, io.BytesIO(b"bad"))
    monkeypatch.setattr(runtime_v3, "P31_ARCHIVE_SIZE", archive.stat().st_size)
    monkeypatch.setattr(runtime_v3, "P31_ARCHIVE_SHA256", sha256(archive))
    monkeypatch.setattr(runtime_v3, "NATIVE_WHEEL_SIZE", 3)
    with pytest.raises(BootstrapError, match="pinned regular native wheel"):
        runtime_v3.validate_native_source(archive)


def test_direct_p31_wheel_ignores_unrelated_notebook_sidecars(tmp_path, monkeypatch):
    import hashlib
    from kaggle_vllm import runtime_v3
    wheel = tmp_path / WHEEL_FILENAME
    wheel.write_bytes(b"pinned native fixture")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    monkeypatch.setattr(runtime_v3, "WHEEL_SHA256", digest)
    (tmp_path / "SHA256SUMS.txt").write_text("\notherhash  unrelated-file.zip\n")
    (tmp_path / "SOURCE_IDENTITY.json").write_text('{"other_notebook_artifact": true}')
    assert runtime_v3.validate_native_source(wheel)["sha256"] == digest
    (tmp_path / "SHA256SUMS.txt").write_text("0" * 64 + "  " + WHEEL_FILENAME + "\n")
    with pytest.raises(BootstrapError, match="checksum manifest mismatch"):
        runtime_v3.validate_native_source(wheel)


@pytest.mark.parametrize("member_type", ["traversal", "absolute", "symlink", "hardlink"])
def test_archive_rejects_unsafe_members(tmp_path, monkeypatch, member_type):
    import io
    import tarfile
    from kaggle_vllm import runtime_v3
    archive = tmp_path / "test.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        name = BUNDLE_NAME + "/metadata/x"
        if member_type == "traversal": name = BUNDLE_NAME + "/../outside"
        if member_type == "absolute": name = "/tmp/outside"
        item = tarfile.TarInfo(name)
        item.size = 1
        if member_type == "symlink":
            item.type, item.linkname, item.size = tarfile.SYMTYPE, "/tmp/outside", 0
        if member_type == "hardlink":
            item.type, item.linkname, item.size = tarfile.LNKTYPE, "/tmp/outside", 0
        tar.addfile(item, io.BytesIO(b"x") if item.size else None)
    monkeypatch.setattr(runtime_v3, "SDK_OWNED_ROOT", tmp_path / "owned")
    monkeypatch.setattr(runtime_v3, "ARCHIVE_SIZE", archive.stat().st_size)
    monkeypatch.setattr(runtime_v3, "ARCHIVE_SHA256", sha256(archive))
    with pytest.raises(BootstrapError, match="unsafe runtime archive member"):
        runtime_v3.extract_runtime(archive, tmp_path / "owned" / "source")
    assert not (tmp_path / "outside").exists()


def test_archive_rejects_outside_sdk_target(tmp_path, monkeypatch):
    from kaggle_vllm import runtime_v3
    monkeypatch.setattr(runtime_v3, "SDK_OWNED_ROOT", tmp_path / "owned")
    with pytest.raises(BootstrapError, match="SDK-owned"):
        runtime_v3.extract_runtime(tmp_path / "missing.tar.gz", tmp_path / "outside")


def test_bootstrap_rejects_incompatible_before_staging(tmp_path, monkeypatch):
    from kaggle_vllm import l4_bootstrap, runtime_v3
    archive = tmp_path / (BUNDLE_NAME + ".tar.gz")
    archive.touch()
    monkeypatch.setattr(runtime_v3, "SDK_OWNED_ROOT", tmp_path / "owned")
    monkeypatch.setattr(l4_bootstrap, "_compatibility", lambda strict: ["wrong ABI"])
    with pytest.raises(BootstrapError, match="wrong ABI"):
        l4_bootstrap.bootstrap(runtime_root=archive, target=tmp_path / "owned")
    assert not (tmp_path / "owned").exists()


def test_runtime_environment_order_and_dedup(tmp_path, monkeypatch):
    from kaggle_vllm.l4_bootstrap import runtime_environment
    pydeps = str(tmp_path / "pydeps")
    monkeypatch.setenv("PYTHONPATH", pydeps + ":/existing:" + pydeps)
    monkeypatch.setenv("LIBRARY_PATH", str(tmp_path / "cuda-driver-link") + ":/other")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/nvidia/lib64:/other")
    env = runtime_environment(tmp_path)
    assert env["PYTHONPATH"] == pydeps + ":/existing"
    assert env["LIBRARY_PATH"] == str(tmp_path / "cuda-driver-link") + ":/other"
    assert env["LD_LIBRARY_PATH"] == "/usr/local/nvidia/lib64:" + pydeps + "/torch/lib:/other"


def test_split_staging_verifies_each_wheel_without_network(tmp_path, monkeypatch):
    import hashlib
    from kaggle_vllm import runtime_v3
    groups = {}
    for group in runtime_v3.SPLIT_GROUPS:
        directory = tmp_path / group
        directory.mkdir()
        groups[group] = directory
    wheel_name = "torchcodec-0.16.0+cu129-py3-none-any.whl"
    wheel_bytes = b"fixture wheel"
    (groups["runtime-core"] / wheel_name).write_bytes(wheel_bytes)
    native = tmp_path / WHEEL_FILENAME
    native.write_bytes(b"fixture native")
    manifest = {"schema_version": 3, "profile": "kaggle-l4x4-cu129", "native": {"filename": WHEEL_FILENAME, "sha256": hashlib.sha256(native.read_bytes()).hexdigest()}, "wheel_count": 1, "wheels": [{"filename": wheel_name, "distribution": "torchcodec", "version": "0.16.0+cu129", "group": "runtime-core", "size": len(wheel_bytes), "sha256": hashlib.sha256(wheel_bytes).hexdigest()}]}
    raw = json.dumps(manifest).encode()
    audit = tmp_path / "audit"
    audit.mkdir()
    audit_raw = json.dumps({"wheel_count": 1, "wheels": [{"filename": wheel_name, "size": len(wheel_bytes), "SHA256": hashlib.sha256(wheel_bytes).hexdigest(), "package_group": "runtime-core"}]}).encode()
    identity_raw = b'{"fixture": true}'
    (audit / "WHEELHOUSE_MANIFEST.json").write_bytes(audit_raw)
    (audit / "SOURCE_IDENTITY.json").write_bytes(identity_raw)
    monkeypatch.setattr(runtime_v3, "SDK_OWNED_ROOT", tmp_path / "owned")
    monkeypatch.setattr(runtime_v3, "MANIFEST_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(runtime_v3, "WHEEL_SHA256", manifest["native"]["sha256"])
    monkeypatch.setattr(runtime_v3, "TORCHCODEC_AUDIT_SHA256", hashlib.sha256(b'{"cuda13_runtime_dependency": false}').hexdigest())
    monkeypatch.setattr(runtime_v3, "WHEELHOUSE_MANIFEST_SHA256", hashlib.sha256(audit_raw).hexdigest())
    monkeypatch.setattr(runtime_v3, "SOURCE_IDENTITY_SHA256", hashlib.sha256(identity_raw).hexdigest())
    monkeypatch.setattr(runtime_v3, "_pinned_metadata", lambda name: raw if name in ("RUNTIME_MANIFEST.json", "L4_INFERENCE_LOCK.json") else b'{"cuda13_runtime_dependency": false}')
    monkeypatch.setattr(runtime_v3, "lean_runtime_manifest", lambda: manifest)
    source = runtime_v3.SplitRuntimeSource(tmp_path, groups, native, audit)
    staged = runtime_v3.stage_split_runtime(source, tmp_path / "owned" / "source")
    assert (staged / "wheelhouse" / wheel_name).read_bytes() == wheel_bytes
    assert runtime_v3.verify_runtime(staged)["wheel_count"] == 1


def test_split_zip_wheel_hash_and_exact_member(tmp_path):
    import hashlib
    import zipfile
    from kaggle_vllm.runtime_v3 import _wheel_source
    name = "packaging-26.3-py3-none-any.whl"
    content = b"pinned fixture wheel"
    record = {"filename": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    archive = tmp_path / "corrected-v2.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(name, content)
    assert _wheel_source(archive, record) == (len(content), record["sha256"])
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("nested/" + name, content)
    with pytest.raises(BootstrapError, match="missing or unsafe"):
        _wheel_source(archive, record)


def test_activation_rejects_preloaded_external_torch(tmp_path, monkeypatch):
    from kaggle_vllm import l4_bootstrap
    target = tmp_path / "target"
    pydeps = target / "pydeps"
    pydeps.mkdir(parents=True)
    manifest = target / "manifests" / "sdk-runtime.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"schema_version": 3, "native_sha256": WHEEL_SHA256, "pydeps": str(pydeps)}))
    monkeypatch.setattr(l4_bootstrap, "_compatibility", lambda strict: [])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__file__="/outside/torch/__init__.py"))
    with pytest.raises(BootstrapError, match="restart the kernel"):
        l4_bootstrap.activate_runtime(target)


def test_doctor_rejects_wrong_driver_same_basename(tmp_path, monkeypatch):
    from kaggle_vllm import doctor_v3
    env = SimpleNamespace(is_kaggle=False, torch=None, torch_cuda=None, cuda_available=False, gpu_count=0, gpus=(), gpu_names=[], gpu_caps=[], cuda_driver=None)
    monkeypatch.setattr(doctor_v3, "collect", lambda: env)
    actual_driver = tmp_path / "real" / "libcuda.so.1"
    actual_driver.parent.mkdir()
    actual_driver.touch()
    fake_driver = tmp_path / "fake" / "libcuda.so.1"
    fake_driver.parent.mkdir()
    fake_driver.touch()
    monkeypatch.setattr(doctor_v3, "DEFAULT_DRIVER", actual_driver)
    alias_dir = tmp_path / "cuda-driver-link"
    alias_dir.mkdir()
    (alias_dir / "libcuda.so").symlink_to(fake_driver)
    report = doctor_v3.doctor_report(target=tmp_path, strict=False)
    alias_check = next(x for x in report["findings"] if x["check"] == "libcuda linker alias")
    assert alias_check["status"] == "error"


def test_fresh_gemma_acceptance_meaning_gate():
    import runpy
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "fresh_kaggle_gemma_v030.py"))
    meaningful = namespace["meaningful_text"]
    assert meaningful("Tensor parallelism splits large tensors across several GPUs so that they work together on one computation.")
    assert not meaningful("\n\n\n")
    assert not meaningful("Hello world, hello world, hello world, hello world, hello world.")
