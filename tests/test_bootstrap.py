from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from kaggle_vllm.bootstrap import activate_runtime, bootstrap, inspect_compatibility
from kaggle_vllm.environment import Environment, GPUInfo
from kaggle_vllm.exceptions import BootstrapError
from kaggle_vllm.profiles import load_profile


def validated_environment() -> Environment:
    return Environment(
        is_kaggle=True,
        python="3.12.13",
        platform="Linux-x86_64",
        torch="2.10.0+cu128",
        torch_path=None,
        torch_cuda="12.8",
        cuda_available=True,
        gpus=(
            GPUInfo(0, "Tesla T4", (7, 5)),
            GPUInfo(1, "Tesla T4", (7, 5)),
        ),
        nccl="2.27.5",
        nvcc="/usr/local/cuda/bin/nvcc",
        nvcc_version="Cuda compilation tools, release 12.8, V12.8.93",
        cuda_home="/usr/local/cuda",
        cuda_driver="/usr/local/nvidia/lib64/libcuda.so.1",
        cmake_library_path="/usr/local/nvidia/lib64",
    )


def host_findings(version: tuple[int, int], system="Linux", machine="x86_64"):
    return inspect_compatibility(
        load_profile(),
        validated_environment(),
        strict=True,
        implementation="CPython",
        python_version=version,
        system=system,
        machine=machine,
    )


def test_cp311_rejected_for_cp312_native_wheel():
    errors = [finding for finding in host_findings((3, 11)) if finding.status == "error"]
    assert any(finding.check == "Python ABI" for finding in errors)


def test_cp312_linux_x86_64_is_accepted():
    assert all(finding.status == "pass" for finding in host_findings((3, 12)))


@pytest.mark.parametrize("system,machine", [("Darwin", "x86_64"), ("Linux", "aarch64")])
def test_native_profile_rejects_other_platforms(system, machine):
    assert any(
        finding.status == "error"
        for finding in host_findings((3, 12), system=system, machine=machine)
    )


def test_dry_run_does_not_create_user_paths(tmp_path):
    paths = {
        "staged": tmp_path / "staged",
        "overlay": tmp_path / "overlay",
        "cache": tmp_path / "cache",
        "manifest": tmp_path / "runtime.json",
    }
    result = bootstrap(
        **paths,
        dry_run=True,
        strict=True,
        environment=validated_environment(),
        implementation="CPython",
        python_version=(3, 12),
        system="Linux",
        machine="x86_64",
    )
    assert result.plan.compatible
    assert not result.completed
    assert not any(path.exists() for path in paths.values())
    assert all("--no-deps" in command for command in result.plan.commands)


def test_bootstrap_writes_manifest_and_is_idempotent(monkeypatch, tmp_path):
    bootstrap_module = importlib.import_module("kaggle_vllm.bootstrap")
    wheel = tmp_path / load_profile().wheel_filename
    wheel.write_bytes(b"mocked by staging")
    calls = []

    def fake_stage_wheel(_wheel, target, **kwargs):
        calls.append(("wheel", kwargs))
        target = Path(target)
        target.mkdir(parents=True)
        (target / "vllm").mkdir()
        return target

    def fake_stage_overlay(_lock, target, **kwargs):
        calls.append(("overlay", kwargs))
        target = Path(target)
        target.mkdir(parents=True)
        (target / "transformers").mkdir()
        return target

    monkeypatch.setattr(bootstrap_module, "download_wheel", lambda *_args: wheel)
    monkeypatch.setattr(bootstrap_module, "stage_wheel", fake_stage_wheel)
    monkeypatch.setattr(bootstrap_module, "stage_dependency_overlay", fake_stage_overlay)
    arguments = {
        "staged": tmp_path / "staged",
        "overlay": tmp_path / "overlay",
        "cache": tmp_path / "cache",
        "manifest": tmp_path / "runtime.json",
        "strict": True,
        "environment": validated_environment(),
        "implementation": "CPython",
        "python_version": (3, 12),
        "system": "Linux",
        "machine": "x86_64",
        "python_executable": "/usr/bin/python3.12",
    }
    first = bootstrap(**arguments)
    second = bootstrap(**arguments)
    assert first.completed and not first.already_complete
    assert second.completed and second.already_complete
    assert [name for name, _ in calls] == ["wheel", "overlay"]
    manifest = json.loads((tmp_path / "runtime.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "kaggle-t4x2-cu128"
    assert manifest["wheel"]["hf_revision"] == load_profile().hf_revision
    assert set(manifest["runtime_environment"]) == {
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "PATH",
    }


def test_bootstrap_refuses_nonempty_runtime_destination(monkeypatch, tmp_path):
    bootstrap_module = importlib.import_module("kaggle_vllm.bootstrap")
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "foreign.txt").write_text("do not overwrite", encoding="utf-8")
    monkeypatch.setattr(
        bootstrap_module,
        "download_wheel",
        lambda *_args: (_ for _ in ()).throw(AssertionError("download not expected")),
    )
    with pytest.raises(BootstrapError, match="refusing to overwrite"):
        bootstrap(
            staged=staged,
            overlay=tmp_path / "overlay",
            cache=tmp_path / "cache",
            manifest=tmp_path / "runtime.json",
            environment=validated_environment(),
            implementation="CPython",
            python_version=(3, 12),
            system="Linux",
            machine="x86_64",
        )


def test_activate_runtime_uses_overlay_before_staged(monkeypatch, tmp_path):
    overlay = tmp_path / "overlay"
    staged = tmp_path / "staged"
    overlay.mkdir()
    staged.mkdir()
    manifest = tmp_path / "runtime.json"
    manifest.write_text(
        json.dumps(
            {
                "paths": {"overlay": str(overlay), "staged": str(staged)},
                "runtime_environment": {
                    "PYTHONPATH": f"{overlay}{os.pathsep}{staged}",
                    "LD_LIBRARY_PATH": "/torch/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64",
                    "PATH": f"{staged / 'bin'}:/usr/bin",
                },
            }
        ),
        encoding="utf-8",
    )
    original_path = list(sys.path)
    monkeypatch.setattr(sys, "path", original_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    assert activate_runtime(manifest)
    assert sys.path[:2] == [str(overlay), str(staged)]
    assert os.environ["PYTHONPATH"].split(os.pathsep)[:2] == [
        str(overlay),
        str(staged),
    ]
    assert os.environ["PATH"].split(os.pathsep)[0] == str(staged / "bin")
