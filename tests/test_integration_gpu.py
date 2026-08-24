import pytest

from kaggle_vllm.environment import collect


@pytest.mark.gpu
@pytest.mark.kaggle
def test_documented_kaggle_gpu_profile():
    environment = collect()
    if not environment.is_kaggle or environment.gpu_count != 2:
        pytest.skip("requires the documented Kaggle dual-GPU runtime")
    assert all(gpu.name == "Tesla T4" for gpu in environment.gpus)
    assert all(gpu.capability == (7, 5) for gpu in environment.gpus)
