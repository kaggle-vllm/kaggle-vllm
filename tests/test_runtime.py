import pytest

from kaggle_vllm.exceptions import RuntimeValidationError
from kaggle_vllm.runtime import validate_tensor_parallel_size


def test_tp_size_accepts_visible_gpu_count():
    validate_tensor_parallel_size(2, gpu_count=2)


def test_tp_two_fails_with_one_gpu():
    with pytest.raises(RuntimeValidationError, match="found 1"):
        validate_tensor_parallel_size(2, gpu_count=1)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_tp_size_must_be_positive_integer(value):
    with pytest.raises(RuntimeValidationError):
        validate_tensor_parallel_size(value, gpu_count=2)
