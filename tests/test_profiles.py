from importlib import resources

from kaggle_vllm.profiles import DEFAULT_PROFILE, load_profile


def test_packaged_profile_identity_and_overlay():
    profile = load_profile()
    assert profile.name == DEFAULT_PROFILE == "kaggle-t4x2-cu128"
    assert profile.python_abi == "cp312"
    assert profile.wheel_filename == (
        "vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-"
        "cp312-cp312-linux_x86_64.whl"
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
