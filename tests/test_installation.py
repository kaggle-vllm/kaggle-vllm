from types import SimpleNamespace

import pytest

from kaggle_vllm.exceptions import InstallationError
from kaggle_vllm.installation import (
    build_runtime_environment,
    stage_dependency_overlay,
    stage_wheel,
)


def test_stage_wheel_uses_target_and_no_deps(monkeypatch, tmp_path):
    wheel = tmp_path / "vllm.whl"
    wheel.write_bytes(b"wheel")
    calls = []
    monkeypatch.setattr(
        "kaggle_vllm.installation.subprocess.run",
        lambda command, check: calls.append((command, check)) or SimpleNamespace(returncode=0),
    )
    target = stage_wheel(wheel, tmp_path / "staged", python_executable="python")
    command = calls[0][0]
    assert target == (tmp_path / "staged").resolve()
    assert "--target" in command
    assert "--no-deps" in command
    assert command[-1] == str(wheel.resolve())


def test_overlay_rejects_torch_requirements(tmp_path):
    requirements = tmp_path / "lock.txt"
    requirements.write_text("transformers==4.57.6\ntorch==2.10.0\n", encoding="utf-8")
    with pytest.raises(InstallationError, match="must not contain torch"):
        stage_dependency_overlay(requirements, tmp_path / "overlay")


def test_runtime_environment_orders_overlay_before_staged(tmp_path):
    environment = build_runtime_environment(
        tmp_path / "staged",
        tmp_path / "overlay",
        torch_library=tmp_path / "torch-lib",
        environ={"PYTHONPATH": "/existing", "LD_LIBRARY_PATH": "/existing-lib"},
    )
    assert environment["PYTHONPATH"].split(":") == [
        str((tmp_path / "overlay").resolve()),
        str((tmp_path / "staged").resolve()),
        "/existing",
    ]
    assert environment["LD_LIBRARY_PATH"].startswith(str((tmp_path / "torch-lib").resolve()))
