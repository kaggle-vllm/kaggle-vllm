"""Generate the output-free target-only v0.3.0 Kaggle acceptance notebook."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle-notebooks/kaggle_vllm_0_3_0_l4x4_final_acceptance.ipynb"


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def main() -> None:
    steps = [
        "0. Immutable Kaggle Input inventory",
        "1. CPython 3.12, Linux x86_64, exactly four NVIDIA L4/SM89 devices",
        "2. Small SDK wheel SHA256",
        "3. Exact corrected split-v2 Dataset discovery",
        "4. Attached P31 native-wheel or final-v2 tar discovery",
        "5. Audit metadata and every pinned runtime/native SHA256",
        "6. Offline small-SDK installation",
        "7. Strict split-v2 offline bootstrap",
        "8. SDK-owned cu129 environment activation",
        "9. Torch 2.13.0+cu129, CUDA 12.9, four L4/SM89 devices",
        "10. Strict doctor JSON and native imports",
        "11. Upstream vLLM 0.30.0 high-level import",
        "12. Native vLLM core import",
        "13. Built-in native FA2 import",
        "14. SDK-owned libcuda -lcuda link probe",
        "15. NCCL world_size=4 subprocess test",
        "16. Offline Gemma 4 QAT model version discovery",
        "17. KaggleLLM.chat semantic TP=4 generation",
        "18. Gemma TP worker cleanup",
        "19. CORE_RELEASE_PASS and FA2_MODEL_RUNTIME_PASS matrix",
        "20. Downloadable evidence ZIP",
    ]
    intro = "# kaggle-vllm 0.3.0 fresh L4x4 split-v2 acceptance\n\nRun in a **brand-new** Kaggle four-L4 competition session with Internet OFF. Attach the small acceptance Dataset, the already-uploaded corrected runtime-core-v2, runtime-large-v2, torch-cu129-v2, wheelhouse-audit-v2 Datasets, the saved P31 Notebook Output, and the Kaggle-provided Gemma 4 QAT Input Model. The small package carries only two additional pinned pure-Python runtime wheels. No 6.6 GB runtime-v3 upload is needed.\n\nThis notebook runs Gemma only. Unrelated-model FA2 inference belongs in a later permitted environment. It does not rebuild vLLM, download model files, or publish anything.\n"
    outline = "## Runner gates\n\n" + "\n".join(f"- STEP {x}" for x in steps) + "\n\nEvery child gate streams to a saved log, has a timeout, runs in its own process group, and is cleaned on error. Real-model FA2 inference is not part of this competition acceptance.\n"
    cell = '''from pathlib import Path
import runpy
import sys

assert sys.version_info[:2] == (3, 12), "Use the Kaggle CPython 3.12 target runtime"
INPUT = Path("/kaggle/input/datasets/waqasm86")
RUNNERS = sorted(INPUT.rglob("production_acceptance_v030.py"))
assert len(RUNNERS) == 1, f"Expected one attached acceptance runner, found {len(RUNNERS)}"

runner = runpy.run_path(str(RUNNERS[0]))["main"]
exit_code = runner([])
assert exit_code == 0, "Core acceptance failed; download and inspect the evidence bundle"
'''
    final = "## Evidence and interpretation\n\nDownload `/kaggle/working/kaggle-vllm-v030-production-acceptance/20_evidence_bundle.zip` and retain the notebook run log. `CORE_RELEASE_PASS` requires every mandatory split-input/Gemma gate. `FA2_MODEL_RUNTIME_PASS` remains false: this competition notebook checks the built-in FA2 binary and import, not unrelated-model inference. P31 Gemma selected `TRITON_ATTN`. Inspect evidence and finish manual redistribution review before any publication.\n"
    notebook = {"cells": [markdown(intro), markdown(outline), code(cell), markdown(final)], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(notebook, indent=2) + "\n")
    print(TARGET)


if __name__ == "__main__":
    main()
