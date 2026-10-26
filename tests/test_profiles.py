import json
import os
import subprocess
import sys
import tomllib
from importlib import resources
from pathlib import Path

import pytest

from kaggle_vllm.dependencies import load_dependency_baseline
from kaggle_vllm.exceptions import ProfileError
from kaggle_vllm.profiles import DEFAULT_PROFILE, BootstrapProfile, load_profile


ROOT = Path(__file__).resolve().parents[1]


def test_project_and_active_profile_are_cp313_pending_t4():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["version"] == "0.2.0"
    assert project["project"]["requires-python"] == ">=3.13,<3.14"
    assert set(project["project"]["dependencies"]) == {"packaging>=23"}
    profile = load_profile()
    assert profile.name == DEFAULT_PROFILE == "kaggle-t4x2-cu128"
    assert profile.status == "target_unvalidated"
    assert profile.python_abi == "cp313"
    assert profile.python_major == 3 and profile.python_minor == 13
    assert profile.gpu_count == 2 and profile.compute_capability == (7, 5)
    assert profile.gpu_name == "NVIDIA Tesla T4"
    assert profile.wheel_status == "pending_kaggle_build"
    assert profile.wheel_filename is None and profile.wheel_sha256 is None
    assert profile.hf_revision is None and profile.overlay_lock is None
    assert not profile.native_ready


def test_historical_profile_is_immutable_parseable_evidence():
    historical = json.loads((ROOT / "compat/historical-py312-v0.2.0/profile.json").read_text())
    assert historical["python"]["abi"] == "cp312"
    assert historical["status"] == "functionally_validated"
    assert historical["wheel"]["filename"].endswith("cp312-cp312-linux_x86_64.whl")
    assert historical["wheel"]["sha256"] == "5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c"


def test_active_profile_rejects_cp312_wheel_registration():
    resource = resources.files("kaggle_vllm").joinpath("profiles", DEFAULT_PROFILE, "profile.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    data["status"] = "functionally_validated"
    data["wheel"].update(status="validated", filename="vllm-0.18.1-cp312-cp312-linux_x86_64.whl",
                         sha256="a" * 64, hf_repo_id="example/native", hf_revision="b" * 40)
    data["overlay"]["lock"] = "validated-lock.txt"
    with pytest.raises(ProfileError, match="filename ABI"):
        BootstrapProfile.from_mapping(data)


def test_profile_rejects_malformed_compute_capability():
    resource = resources.files("kaggle_vllm").joinpath("profiles", DEFAULT_PROFILE, "profile.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    data["runtime"]["compute_capability"] = ["7", 5]
    with pytest.raises(ProfileError, match="two integers"):
        BootstrapProfile.from_mapping(data)


def test_provisional_dependency_contract_exposes_transformers_conflict():
    specs = {x.distribution: x for x in load_dependency_baseline()}
    assert specs["transformers"].specifier == ">=4.56.0,<5"
    assert specs["transformers"].validated_version is None
    assert specs["flashinfer-python"].required is False


def test_active_profile_contains_no_later_gpu_generation():
    active = [ROOT / "pyproject.toml", ROOT / "compat/kaggle-t4x2-cu128.json"]
    active.extend((ROOT / "src/kaggle_vllm/profiles/kaggle-t4x2-cu128").glob("*"))
    forbidden = ("0.3.0", "NVIDIA L4", "SM89", "TP=4", "cu129", "CUDA 12.9")
    assert all(marker not in path.read_text() for path in active for marker in forbidden)


def test_import_has_no_native_import_network_or_install_side_effect(tmp_path):
    code = """import socket, subprocess, sys
socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network'))
subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError('install'))
import kaggle_vllm
assert kaggle_vllm.__version__ == '0.2.0'
assert 'vllm' not in sys.modules
"""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
