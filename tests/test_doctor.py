from kaggle_vllm.doctor import suggested_build_env


def test_kaggle_build_env_contains_cuda():
    env = suggested_build_env()
    assert env["CUDA_HOME"] == "/usr/local/cuda"
    assert env["CUDAToolkit_ROOT"] == "/usr/local/cuda"
    assert env["TORCH_CUDA_ARCH_LIST"] == "7.5"
    assert env["MAX_JOBS"] == "1"
    assert env["NVCC_THREADS"] == "1"
