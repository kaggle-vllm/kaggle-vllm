import hashlib

import pytest

from kaggle_vllm.checksums import sha256_file, verify_sha256
from kaggle_vllm.exceptions import ChecksumMismatchError


def test_checksum_calculation_and_verification(tmp_path):
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"validated artifact")
    expected = hashlib.sha256(b"validated artifact").hexdigest()
    assert sha256_file(artifact) == expected
    assert verify_sha256(artifact, expected) == expected


def test_checksum_mismatch_raises(tmp_path):
    artifact = tmp_path / "artifact.whl"
    artifact.write_bytes(b"wrong")
    with pytest.raises(ChecksumMismatchError):
        verify_sha256(artifact, "0" * 64)
