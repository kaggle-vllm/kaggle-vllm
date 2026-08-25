from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from kaggle_vllm.bootstrap import (
    activate_runtime,
    bootstrap,
    build_reset_plan,
    execute_reset,
    inspect_compatibility,
)
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


def write_reset_manifest(
    manifest: Path,
    *,
    staged: Path,
    overlay: Path,
    cache: Path,
) -> None:
    profile = load_profile()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": profile.name,
                "wheel": {
                    "sha256": profile.wheel_sha256,
                    "hf_revision": profile.hf_revision,
                },
                "paths": {
                    "staged": str(staged.resolve()),
                    "overlay": str(overlay.resolve()),
                    "cache": str(cache.resolve()),
                    "manifest": str(manifest.resolve()),
                }
            }
        ),
        encoding="utf-8",
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


def test_reset_dry_run_deletes_nothing(tmp_path):
    staged = tmp_path / "staged"
    overlay = tmp_path / "overlay"
    cache = tmp_path / "cache"
    manifest = tmp_path / "runtime.json"
    for path in (staged, overlay, cache):
        path.mkdir()
        (path / "marker.txt").write_text(path.name, encoding="utf-8")
    write_reset_manifest(
        manifest,
        staged=staged,
        overlay=overlay,
        cache=cache,
    )

    result = bootstrap(
        staged=staged,
        overlay=overlay,
        cache=cache,
        manifest=manifest,
        reset_runtime=True,
        dry_run=True,
        environment=validated_environment(),
        implementation="CPython",
        python_version=(3, 12),
        system="Linux",
        machine="x86_64",
    )

    assert result.reset_plan is not None
    assert result.reset_plan.safe
    assert not result.reset_completed
    assert all(path.exists() for path in (staged, overlay, cache, manifest))
    reset_data = result.to_dict()["reset"]
    assert reset_data["safe"] is True
    assert reset_data["completed"] is False
    assert next(
        target for target in reset_data["targets"] if target["label"] == "cache"
    )["action"] == "preserve"


def test_execute_reset_removes_only_owned_runtime_and_preserves_cache(tmp_path):
    staged = tmp_path / "staged"
    overlay = tmp_path / "overlay"
    cache = tmp_path / "cache"
    manifest = tmp_path / "runtime.json"
    unrelated = tmp_path / "unrelated"
    for path in (staged, overlay, cache, unrelated):
        path.mkdir()
        (path / "marker.txt").write_text(path.name, encoding="utf-8")
    write_reset_manifest(
        manifest,
        staged=staged,
        overlay=overlay,
        cache=cache,
    )

    removed = execute_reset(
        build_reset_plan(
            staged=staged,
            overlay=overlay,
            cache=cache,
            manifest=manifest,
        )
    )

    assert set(removed) == {staged.resolve(), overlay.resolve(), manifest.resolve()}
    assert not staged.exists()
    assert not overlay.exists()
    assert not manifest.exists()
    assert (cache / "marker.txt").read_text(encoding="utf-8") == "cache"
    assert (unrelated / "marker.txt").read_text(encoding="utf-8") == "unrelated"


@pytest.mark.parametrize(
    "dangerous",
    [
        Path("/"),
        Path("/kaggle"),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("/home"),
        Path("/usr"),
        Path("/usr/local"),
        Path("/opt"),
        Path("/etc"),
        Path("/var"),
        Path("/tmp"),
    ],
)
def test_reset_rejects_dangerous_staged_paths(dangerous, tmp_path):
    with pytest.raises(BootstrapError, match="refusing dangerous"):
        build_reset_plan(
            staged=dangerous,
            overlay=tmp_path / "overlay",
            cache=tmp_path / "cache",
            manifest=tmp_path / "runtime.json",
        )


def test_reset_rejects_home_directory(tmp_path):
    with pytest.raises(BootstrapError, match="refusing dangerous"):
        build_reset_plan(
            staged=Path.home(),
            overlay=tmp_path / "overlay",
            cache=tmp_path / "cache",
            manifest=tmp_path / "runtime.json",
        )


@pytest.mark.parametrize("dangerous", ["", Path.cwd()])
def test_reset_rejects_empty_or_current_directory(dangerous, tmp_path):
    with pytest.raises(BootstrapError, match="refusing"):
        build_reset_plan(
            staged=dangerous,
            overlay=tmp_path / "overlay",
            cache=tmp_path / "cache",
            manifest=tmp_path / "runtime.json",
        )


def test_reset_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "staged-link"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(BootstrapError, match="traverses a symlink"):
        build_reset_plan(
            staged=link,
            overlay=tmp_path / "overlay",
            cache=tmp_path / "cache",
            manifest=tmp_path / "runtime.json",
        )
    assert outside.is_dir()


def test_reset_rejects_mismatching_manifest_paths(tmp_path):
    staged = tmp_path / "staged"
    overlay = tmp_path / "overlay"
    cache = tmp_path / "cache"
    manifest = tmp_path / "runtime.json"
    staged.mkdir()
    overlay.mkdir()
    write_reset_manifest(
        manifest,
        staged=tmp_path / "different-staged",
        overlay=overlay,
        cache=cache,
    )

    with pytest.raises(BootstrapError, match="manifest identity or paths do not match"):
        build_reset_plan(
            staged=staged,
            overlay=overlay,
            cache=cache,
            manifest=manifest,
        )


def test_reset_rejects_mismatching_manifest_identity(tmp_path):
    staged = tmp_path / "staged"
    overlay = tmp_path / "overlay"
    cache = tmp_path / "cache"
    manifest = tmp_path / "runtime.json"
    staged.mkdir()
    overlay.mkdir()
    write_reset_manifest(
        manifest,
        staged=staged,
        overlay=overlay,
        cache=cache,
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["wheel"]["hf_revision"] = "different"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(BootstrapError, match="manifest identity or paths do not match"):
        build_reset_plan(
            staged=staged,
            overlay=overlay,
            cache=cache,
            manifest=manifest,
        )


def test_reset_rejects_unowned_custom_nonempty_path(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "user.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(BootstrapError, match="cannot prove ownership"):
        build_reset_plan(
            staged=staged,
            overlay=tmp_path / "overlay",
            cache=tmp_path / "cache",
            manifest=tmp_path / "runtime.json",
        )


def test_reset_rejects_parent_of_selected_resource(tmp_path):
    staged = tmp_path / "runtime"
    with pytest.raises(BootstrapError, match="overlaps"):
        build_reset_plan(
            staged=staged,
            overlay=staged / "overlay",
            cache=tmp_path / "cache",
            manifest=tmp_path / "runtime.json",
        )


def test_reset_rejects_runtime_nested_inside_cache(tmp_path):
    cache = tmp_path / "cache"
    with pytest.raises(BootstrapError, match="overlaps"):
        build_reset_plan(
            staged=cache / "staged",
            overlay=tmp_path / "overlay",
            cache=cache,
            manifest=tmp_path / "runtime.json",
        )


def test_reset_missing_custom_paths_is_idempotent(tmp_path):
    plan = build_reset_plan(
        staged=tmp_path / "staged",
        overlay=tmp_path / "overlay",
        cache=tmp_path / "cache",
        manifest=tmp_path / "runtime.json",
    )
    assert execute_reset(plan) == ()
    assert execute_reset(plan) == ()
    assert all(target.action in {"absent", "preserve"} for target in plan.targets)


def test_reset_handles_partial_owned_state(tmp_path):
    staged = tmp_path / "staged"
    overlay = tmp_path / "overlay"
    cache = tmp_path / "cache"
    manifest = tmp_path / "runtime.json"
    overlay.mkdir()
    cache.mkdir()
    write_reset_manifest(
        manifest,
        staged=staged,
        overlay=overlay,
        cache=cache,
    )

    removed = execute_reset(
        build_reset_plan(
            staged=staged,
            overlay=overlay,
            cache=cache,
            manifest=manifest,
        )
    )
    assert set(removed) == {overlay.resolve(), manifest.resolve()}
    assert cache.exists()


def test_reset_partial_failure_reports_prior_removals(monkeypatch, tmp_path):
    bootstrap_module = importlib.import_module("kaggle_vllm.bootstrap")
    staged = tmp_path / "staged"
    overlay = tmp_path / "overlay"
    cache = tmp_path / "cache"
    manifest = tmp_path / "runtime.json"
    staged.mkdir()
    overlay.mkdir()
    cache.mkdir()
    write_reset_manifest(
        manifest,
        staged=staged,
        overlay=overlay,
        cache=cache,
    )
    plan = build_reset_plan(
        staged=staged,
        overlay=overlay,
        cache=cache,
        manifest=manifest,
    )
    original_rmtree = bootstrap_module.shutil.rmtree

    def fail_on_overlay(path):
        if path == overlay.resolve():
            raise OSError("simulated overlay failure")
        original_rmtree(path)

    monkeypatch.setattr(bootstrap_module.shutil, "rmtree", fail_on_overlay)
    with pytest.raises(BootstrapError, match=r"overlay.*already removed:.*staged"):
        execute_reset(plan)
    assert not staged.exists()
    assert overlay.exists()
    assert manifest.exists()
    assert cache.exists()


def test_bootstrap_reset_requires_yes(tmp_path):
    paths = {
        "staged": tmp_path / "staged",
        "overlay": tmp_path / "overlay",
        "cache": tmp_path / "cache",
        "manifest": tmp_path / "runtime.json",
    }
    for label in ("staged", "overlay"):
        paths[label].mkdir()
    write_reset_manifest(**paths)

    with pytest.raises(BootstrapError, match="requires explicit --yes"):
        bootstrap(
            **paths,
            reset_runtime=True,
            environment=validated_environment(),
            implementation="CPython",
            python_version=(3, 12),
            system="Linux",
            machine="x86_64",
        )
    assert all(paths[label].exists() for label in ("staged", "overlay", "manifest"))


def test_confirmed_reset_then_bootstrap_recreates_runtime(monkeypatch, tmp_path):
    bootstrap_module = importlib.import_module("kaggle_vllm.bootstrap")
    paths = {
        "staged": tmp_path / "staged",
        "overlay": tmp_path / "overlay",
        "cache": tmp_path / "cache",
        "manifest": tmp_path / "runtime.json",
    }
    for label in ("staged", "overlay", "cache"):
        paths[label].mkdir()
        (paths[label] / "old.txt").write_text(label, encoding="utf-8")
    write_reset_manifest(**paths)
    wheel = tmp_path / load_profile().wheel_filename
    wheel.write_bytes(b"mocked by staging")

    def fake_stage_wheel(_wheel, target, **_kwargs):
        target.mkdir(parents=True)
        (target / "vllm").mkdir()
        return target

    def fake_stage_overlay(_lock, target, **_kwargs):
        target.mkdir(parents=True)
        (target / "transformers").mkdir()
        return target

    monkeypatch.setattr(bootstrap_module, "download_wheel", lambda *_args: wheel)
    monkeypatch.setattr(bootstrap_module, "stage_wheel", fake_stage_wheel)
    monkeypatch.setattr(bootstrap_module, "stage_dependency_overlay", fake_stage_overlay)

    result = bootstrap(
        **paths,
        reset_runtime=True,
        yes=True,
        strict=True,
        environment=validated_environment(),
        implementation="CPython",
        python_version=(3, 12),
        system="Linux",
        machine="x86_64",
        python_executable="/usr/bin/python3.12",
    )

    assert result.completed and result.reset_completed
    assert not result.already_complete
    assert (paths["staged"] / "vllm").is_dir()
    assert (paths["overlay"] / "transformers").is_dir()
    assert not (paths["staged"] / "old.txt").exists()
    assert not (paths["overlay"] / "old.txt").exists()
    assert (paths["cache"] / "old.txt").read_text(encoding="utf-8") == "cache"
    assert paths["manifest"].is_file()


def test_yes_without_reset_is_rejected(tmp_path):
    with pytest.raises(BootstrapError, match="only valid together"):
        bootstrap(
            staged=tmp_path / "staged",
            overlay=tmp_path / "overlay",
            cache=tmp_path / "cache",
            manifest=tmp_path / "runtime.json",
            yes=True,
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
