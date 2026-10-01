"""Replay reviewed CPU-only notebook analysis cells into a separate output folder.

Requires pandas and matplotlib. Never runs setup, download, server or GPU cells.
Historical plots are regenerated from the same values; PNG byte identity depends
on the original unrecorded plotting-library/font versions.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

from verify_allam_evidence import EVIDENCE, read_json, validate


def reproduce(output):
    import matplotlib
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validate()
    assert output.resolve() != EVIDENCE.resolve()
    for stage, dirname, cells in [
        ("p3", "allam-characterization", [18, 20, 22, 24, 26, 28]),
        ("p4", "allam-p4-final", [18, 20, 22, 24, 26]),
    ]:
        source = EVIDENCE / stage / dirname
        dest = output / stage
        evidence, plots = dest / "evidence", dest / "plots"
        evidence.mkdir(parents=True, exist_ok=True)
        plots.mkdir(parents=True, exist_ok=True)
        notebook = read_json(next((EVIDENCE / stage).glob("*.ipynb")))
        fertility = read_json(source / "evidence/tokenizer-fertility.json")
        matrix = read_json(
            source
            / "evidence"
            / ("matrix-summary.json" if stage == "p3" else "server-matrix.json")
        )
        scope = {
            "pd": pd,
            "json": json,
            "EVIDENCE_DIR": evidence,
            "PLOTS_DIR": plots,
            "matrix_summaries": matrix,
            "matrix_results": matrix,
            "fertility_df": pd.DataFrame(fertility),
            "display": lambda *_: None,
            "LANGUAGES": ["english", "arabic"],
            "CONCURRENCY_LEVELS": [1, 2, 4, 8],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            for index in cells:
                code = "".join(notebook["cells"][index]["source"])
                # Only reviewed analysis cells from the validated immutable notebook.
                exec(compile(code, f"{stage}:analysis-cell-{index}", "exec"), scope)  # noqa: S102
        for path in evidence.glob("*.csv"):
            original = pd.read_csv(source / "evidence" / path.name)
            replay = pd.read_csv(path)
            pd.testing.assert_frame_equal(
                original, replay, check_dtype=False, rtol=1e-9, atol=1e-10
            )
        original_findings = read_json(source / "evidence/deterministic-findings.json")

        # Tiny float serialization differences are tolerated through numeric comparison.
        def compare(a, b):
            from verify_allam_evidence import close

            if isinstance(a, dict):
                assert a.keys() == b.keys()
                for key in a:
                    compare(a[key], b[key])
            elif isinstance(a, list):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    compare(x, y)
            elif isinstance(a, float):
                close(a, b)
            else:
                assert a == b, (a, b)

        compare(original_findings, read_json(evidence / "deterministic-findings.json"))
        assert {p.name for p in plots.glob("*.png")} == {
            p.name for p in (source / "plots").glob("*.png")
        }
        plt.close("all")
        print(
            f"{stage}: all CSV/JSON analysis reproduced; {len(list(plots.glob('*.png')))} plots regenerated"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    reproduce(parser.parse_args().output)
