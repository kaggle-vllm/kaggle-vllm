import json
from pathlib import Path

from research.analysis.generate_figures import incomplete_m4_status


def test_partial_m4_progress_is_not_promoted_to_final_analysis(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "principal_planned_shards": 60,
                "principal_shards": {
                    "qwen25_3b-short-r00": {
                        "status": "PRINCIPAL_SHARD_PRESERVED"
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert incomplete_m4_status(evidence) == (
        "INCOMPLETE_M4_PRINCIPAL_EVIDENCE_1_OF_60_SHARDS;"
        "QWEN25_3B-SHORT=1_OF_5_REPETITIONS"
    )


def test_absent_m4_evidence_remains_explicitly_incomplete(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"principal_shards": {}}), encoding="utf-8")
    assert incomplete_m4_status(evidence) == "INCOMPLETE_NO_M4_PRINCIPAL_EVIDENCE"
