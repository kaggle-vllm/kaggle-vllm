"""CPU-only development checks."""
import sys
import pytest

from kaggle_vllm.runtime import check_runtime

def test_python_version_guard():
    if sys.version_info[:2] != (3, 12):
        with pytest.raises(
            RuntimeError, match="python3.12"
        ):
            check_runtime()
