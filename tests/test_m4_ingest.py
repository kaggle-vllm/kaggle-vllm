import json
import zipfile
from pathlib import Path

import pytest

from kaggle_vllm.research.errors import ResearchEvidenceError
from kaggle_vllm.research.m4_ingest import inspect_zip, notebook_sources, verify_runtime


def test_zip_audit_rejects_traversal(tmp_path: Path) -> None:
    candidate = tmp_path / "bad.zip"
    with zipfile.ZipFile(candidate, "w") as archive:
        archive.writestr("../escape", "bad")
    with pytest.raises(ResearchEvidenceError, match="unsafe ZIP path"):
        inspect_zip(candidate)


def test_notebook_source_comparison_ignores_outputs(tmp_path: Path) -> None:
    first = tmp_path / "first.ipynb"
    second = tmp_path / "second.ipynb"
    base = {"cells": [{"cell_type": "code", "id": "cell", "source": ["x = 1\n"], "outputs": []}]}
    first.write_text(json.dumps(base))
    base["cells"][0]["outputs"] = [{"output_type": "stream", "text": ["ok\n"]}]
    second.write_text(json.dumps(base))
    assert notebook_sources(first) == notebook_sources(second)


def test_runtime_gate_requires_two_sm75_t4s() -> None:
    runtime = {
        "profile": "kaggle-t4x2-cu128",
        "strict": True,
        "wheel": {
            "sha256": "5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c",
            "hf_repo_id": "waqasm86/kaggle-vllm-binaries",
            "hf_revision": "f6b4f10de54924ed6fe9e28cceab84eca7276ab6",
        },
        "environment": {
            "python": "3.12.13",
            "torch": "2.10.0+cu128",
            "torch_cuda": "12.8",
            "cuda_available": True,
            "nccl": "2.27.5",
            "driver_version": "580.159.04",
            "nvcc_version": "Cuda compilation tools, release 12.8, V12.8.93",
            "gpus": [
                {"name": "Tesla T4", "capability": [7, 5]},
                {"name": "Tesla T4", "capability": [7, 5]},
            ],
        },
    }
    verify_runtime(runtime)
    runtime["environment"]["gpus"][1]["capability"] = [8, 0]
    with pytest.raises(ResearchEvidenceError, match="runtime identity mismatch"):
        verify_runtime(runtime)
