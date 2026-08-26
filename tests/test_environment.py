from __future__ import annotations

from types import SimpleNamespace

from kaggle_vllm.environment import GPUInfo, collect, detect_kaggle
from kaggle_vllm.runtime import all_sm75, all_tesla_t4


class FakeCuda:
    def is_available(self):
        return True

    def device_count(self):
        return 2

    def get_device_name(self, index):
        return "Tesla T4"

    def get_device_capability(self, index):
        return (7, 5)

    def get_device_properties(self, index):
        return SimpleNamespace(total_memory=15_636_037_632)

    nccl = SimpleNamespace(version=lambda: (2, 27, 5))


def test_kaggle_detection_from_environment(tmp_path):
    assert detect_kaggle({"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, tmp_path)
    assert not detect_kaggle({}, tmp_path)


def test_collect_detects_two_tesla_t4_sm75(monkeypatch):
    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        __file__="/system/torch/__init__.py",
        version=SimpleNamespace(cuda="12.8"),
        cuda=FakeCuda(),
    )
    monkeypatch.setattr("kaggle_vllm.environment.detect_kaggle", lambda: True)
    environment = collect(fake_torch)
    assert environment.gpu_count == 2
    assert environment.gpu_names == ["Tesla T4", "Tesla T4"]
    assert environment.gpu_caps == [(7, 5), (7, 5)]
    assert environment.nccl == "2.27.5"
    assert all_tesla_t4(environment.gpus)
    assert all_sm75(environment.gpus)


def test_t4_and_sm75_checks_require_nonempty_devices():
    assert not all_tesla_t4(())
    assert not all_sm75(())
    assert not all_tesla_t4((GPUInfo(0, "A100", (8, 0)),))
    assert not all_sm75((GPUInfo(0, "Tesla T4", (8, 0)),))


def test_collect_names_driver_reported_cuda_max_without_conflating_sm(monkeypatch):
    monkeypatch.setattr(
        "kaggle_vllm.environment.shutil.which",
        lambda command: "/usr/bin/nvidia-smi" if command == "nvidia-smi" else None,
    )
    monkeypatch.setattr(
        "kaggle_vllm.environment._run",
        lambda *_args: "Driver Version: 580.159.04     CUDA Version: 13.0",
    )
    environment = collect(
        SimpleNamespace(
            __version__="2.10.0+cu128",
            __file__="/system/torch/__init__.py",
            version=SimpleNamespace(cuda="12.8"),
            cuda=FakeCuda(),
        )
    )
    assert environment.driver_version == "580.159.04"
    assert environment.driver_reported_cuda_max == "13.0"
    assert environment.gpu_caps == [(7, 5), (7, 5)]
