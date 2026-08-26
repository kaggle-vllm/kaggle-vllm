import json
from importlib import resources

import pytest

from kaggle_vllm.dependencies import load_dependency_baseline
from kaggle_vllm.exceptions import ProfileError
from kaggle_vllm.profiles import DEFAULT_PROFILE, BootstrapProfile, load_profile


def test_packaged_profile_identity_and_overlay():
    profile = load_profile()
    assert profile.name == DEFAULT_PROFILE == "kaggle-t4x2-cu128"
    assert profile.python_abi == "cp312"
    assert profile.wheel_filename == (
        "vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl"
    )
    assert profile.wheel_sha256 == (
        "5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c"
    )
    assert profile.hf_repo_id == "waqasm86/kaggle-vllm-binaries"
    assert profile.hf_revision == "f6b4f10de54924ed6fe9e28cceab84eca7276ab6"
    root = resources.files("kaggle_vllm").joinpath("profiles", profile.name)
    lock = root.joinpath(profile.overlay_lock).read_text(encoding="utf-8")
    assert "transformers==4.57.6" in lock
    assert "\ntorch==" not in f"\n{lock.casefold()}"
    baseline = load_dependency_baseline()
    assert any(
        item.distribution == "xgrammar" and item.validated_version == "0.2.3"
        for item in baseline
    )
    assert any(item.import_name == "tvm_ffi" for item in baseline)


def test_profile_rejects_malformed_compute_capability():
    resource = resources.files("kaggle_vllm").joinpath(
        "profiles", DEFAULT_PROFILE, "profile.json"
    )
    data = json.loads(resource.read_text(encoding="utf-8"))
    data["runtime"]["compute_capability"] = ["7", 5]
    with pytest.raises(ProfileError, match="two integers"):
        BootstrapProfile.from_mapping(data)
