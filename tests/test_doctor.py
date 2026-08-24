from kaggle_vllm.doctor import profile_failures, suggested_build_env
from kaggle_vllm.environment import Environment, GPUInfo


def validated_environment() -> Environment:
    return Environment(
        is_kaggle=True,
        python="3.12.13",
        platform="Linux",
        torch="2.10.0+cu128",
        torch_path="/usr/local/lib/python3.12/dist-packages/torch/__init__.py",
        torch_cuda="12.8",
        cuda_available=True,
        gpus=(GPUInfo(0, "Tesla T4", (7, 5)), GPUInfo(1, "Tesla T4", (7, 5))),
        nccl="2.27.5",
        nvcc="/usr/local/cuda/bin/nvcc",
        nvcc_version="12.8",
        cuda_home="/usr/local/cuda",
        cuda_driver="/usr/local/nvidia/lib64/libcuda.so",
        cmake_library_path="/usr/local/nvidia/lib64",
    )


def test_validated_profile_has_no_failures():
    assert profile_failures(validated_environment()) == []


def test_kaggle_build_env_contains_cuda():
    environment = suggested_build_env()
    assert environment["CUDA_HOME"] == "/usr/local/cuda"
    assert environment["CUDAToolkit_ROOT"] == "/usr/local/cuda"
    assert environment["TORCH_CUDA_ARCH_LIST"] == "7.5"
    assert environment["MAX_JOBS"] == "1"
    assert environment["NVCC_THREADS"] == "1"
