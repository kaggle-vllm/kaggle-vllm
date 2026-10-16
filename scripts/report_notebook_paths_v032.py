"""Inventory manual acceptance notebooks without executing their cells."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "v032-production-release"
PARENT = ROOT.parent
PRIMARY = {
    "0.2.0": "kaggle-vllm-v020-t4x2-public-acceptance.ipynb",
    "0.3.2": "kaggle-vllm-v032-l4x4-public-acceptance.ipynb",
}
HISTORICAL = (
    "kaggle_vllm_0_2_0_",
    "kaggle_vllm_0_3_0_l4x4_",
)


def inspect(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("nbformat") != 4 or not isinstance(data.get("cells"), list):
        raise ValueError(f"invalid notebook structure: {path}")
    for cell in data["cells"]:
        if cell.get("cell_type") not in ("code", "markdown", "raw"):
            raise ValueError(f"invalid notebook cell: {path}")
        if not isinstance(cell.get("source"), (str, list)):
            raise ValueError(f"invalid notebook source: {path}")
        if cell["cell_type"] == "code" and path.name in PRIMARY.values():
            source = cell["source"]
            ast.parse("".join(source) if isinstance(source, list) else source)
    return {
        "filename": path.name,
        "absolute_path": str(path.resolve()),
        "parent_directory": str(path.parent.resolve()),
        "size_bytes": path.stat().st_size,
        "cell_count": len(data["cells"]),
        "structural_validation": "PASS (JSON and Python AST)" if path.name in PRIMARY.values() else "PASS (JSON; historical magics not parsed)",
        "has_saved_execution": any(
            c.get("execution_count") is not None or bool(c.get("outputs"))
            for c in data["cells"] if c["cell_type"] == "code"
        ),
    }


def main() -> None:
    notebook_roots = [ROOT / "kaggle-notebooks", PARENT / "kaggle-vllm" / "kaggle-notebooks"]
    results = {}
    for version, name in PRIMARY.items():
        matches = list(ROOT.rglob(name))
        print(f"\nNOTEBOOK: {name}")
        if len(matches) != 1:
            raise SystemExit(f"expected exactly one worktree notebook, found {len(matches)}")
        path = matches[0]
        print(f"DIRECTORY: {path.parent.resolve()}")
        print(f"ABSOLUTE PATH: {path.resolve()}")
        print(f"EXISTS: {path.is_file()}")
        print(f"SIZE: {path.stat().st_size}")
        result = inspect(path)
        result.update({
            "sdk_version": version,
            "intended_gpu": "NVIDIA Tesla T4 x2 / SM75" if version == "0.2.0" else "NVIDIA L4 x4 / SM89",
            "python_abi": "CPython 3.12",
            "runtime_profile": "kaggle-t4x2-cu128" if version == "0.2.0" else "kaggle-l4x4-cu129",
            "required_kaggle_settings": "matching GPU topology; Internet enabled; Python 3.12",
            "external_inputs_required": False,
            "publication_dependencies": [] if version == "0.2.0" else [
                "complete immutable Hugging Face runtime", "final PyPI 0.3.2 SDK", "GitHub v0.3.2 release", "redistribution clearance"
            ],
            "executed_by_codex": False,
            "manual_acceptance_readiness": "READY FOR MANUAL KAGGLE EXECUTION" if version == "0.2.0" else "BLOCKED — PUBLIC RELEASE INCOMPLETE",
        })
        results[version] = result

    historical = []
    unrelated = []
    for root in notebook_roots:
        for path in sorted(root.rglob("*.ipynb")):
            if path.name in PRIMARY.values():
                continue
            entry = inspect(path)
            if path.name.startswith(HISTORICAL):
                t4 = path.name.startswith("kaggle_vllm_0_2_0_")
                entry.update({
                    "sdk_version": "0.2.0" if t4 else "0.3.0/0.3.1 historical",
                    "intended_gpu": "NVIDIA Tesla T4 x2 / SM75" if t4 else "NVIDIA L4 x4 / SM89",
                    "python_abi": "CPython 3.12 target; historical notebook metadata not independently qualified",
                    "runtime_profile": "kaggle-t4x2-cu128" if t4 else "kaggle-l4x4-cu129",
                    "required_kaggle_settings": "Matching GPU topology; consult historical notebook for Internet and Input settings",
                    "external_inputs_required": "historical notebook-specific; inspect before use",
                    "publication_dependencies": ["historical artifact paths and/or Kaggle Inputs as specified in that notebook"],
                    "executed_by_codex": False,
                    "manual_acceptance_readiness": "NOT PRIMARY PUBLIC ACCEPTANCE",
                })
                entry["purpose"] = "historical T4 or L4 research/acceptance; not primary public acceptance"
                historical.append(entry)
            else:
                unrelated.append({"filename": path.name, "absolute_path": str(path.resolve())})
    payload = {
        "schema_version": 1,
        "worktree": str(ROOT),
        "public_status": {
            "0.2.0": "historical public release; exact 0.2.0 artifact evidence only",
            "0.3.2": "PUBLICATION BLOCKED",
            "github": "v0.3.2 release absent; draft PR #39 open",
            "pypi": "0.3.2 absent at https://pypi.org/project/kaggle-vllm/0.3.2/",
            "hugging_face": "v0.3.2/l4x4 native/metadata namespace absent; no immutable release commit; 168 dependencies pinned to publishers",
        },
        "primary": results,
        "historical_relevant": historical,
        "other_notebooks": unrelated,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "21-manual-notebook-paths.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    t4, l4 = results["0.2.0"], results["0.3.2"]
    handoff = f"""=========================================================
KAGGLE-VLLM — MANUAL NOTEBOOK EXECUTION HANDOFF
=========================================================

RELEASE 0.2.0 — NVIDIA T4 × 2

Notebook:
{t4['filename']}

Local directory:
{t4['parent_directory']}

Absolute notebook path:
{t4['absolute_path']}

Profile:
{t4['runtime_profile']}

Public runtime status:
Historical public 0.2.0 runtime preserved at immutable Hugging Face revision f6b4f10de54924ed6fe9e28cceab84eca7276ab6.

Manual execution status:
NOT EXECUTED BY CODEX

---------------------------------------------------------

RELEASE 0.3.2 — NVIDIA L4 × 4

Notebook:
{l4['filename']}

Local directory:
{l4['parent_directory']}

Absolute notebook path:
{l4['absolute_path']}

Profile:
kaggle-l4x4-cu129

GitHub publication:
BLOCKED — draft PR #39 open; v0.3.2 release absent (https://github.com/kaggle-vllm/kaggle-vllm/pull/39).

PyPI publication:
BLOCKED — 0.3.2 absent (https://pypi.org/project/kaggle-vllm/0.3.2/).

Hugging Face runtime:
BLOCKED — v0.3.2/l4x4 native/metadata namespace absent; no immutable release commit; 168 dependencies pinned to publishers (https://huggingface.co/waqasm86/kaggle-vllm-binaries).

Manual execution status:
NOT EXECUTED BY CODEX

Readiness:
BLOCKED — PUBLIC RELEASE INCOMPLETE

=========================================================
"""
    markdown = "# Manual notebook paths and publication status\n\n```text\n" + handoff + "```\n\n"
    markdown += "Both primary notebooks passed JSON and Python AST structural validation; neither was executed by Codex.\n\n"
    markdown += "## Historical relevant notebooks\n\n"
    markdown += "\n".join(f"- `{x['absolute_path']}` — {x['purpose']} (structure: {x['structural_validation']}; saved execution: {x['has_saved_execution']})" for x in historical)
    markdown += "\n\n## Other notebooks (outside primary acceptance)\n\n"
    markdown += "\n".join(f"- `{x['absolute_path']}`" for x in unrelated) + "\n"
    (REPORTS / "21-manual-notebook-paths.md").write_text(markdown, encoding="utf-8")
    print("\n" + handoff)


if __name__ == "__main__":
    main()
