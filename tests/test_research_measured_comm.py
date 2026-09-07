from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from kaggle_vllm.research.architecture import (
    derive_hidden_state_payload_bytes,
    parse_architecture_metadata,
)
from kaggle_vllm.research.cli import main
from kaggle_vllm.research.comparison import build_model_vs_observed
from kaggle_vllm.research.errors import (
    CommunicationFitError,
    ProvenanceError,
    ResearchEvidenceError,
)
from kaggle_vllm.research.measured_comm import (
    CSV_FIELDS,
    RAW_SCHEMA,
    REQUIRED_PAYLOAD_BYTES,
    AllReduceObservation,
    assert_equivalent_ledgers,
    fit_communication_model,
    load_allreduce_csv,
    load_allreduce_json,
    summarize_allreduce,
    validate_observations,
)
from kaggle_vllm.research.provenance import sha256_file, verify_sha256_manifest
from kaggle_vllm.research.report import write_m3_outputs
from kaggle_vllm.research.resources import require_disk_budget
from scripts.kaggle_measured_allreduce import format_command_capture


def observations(
    payloads: tuple[int, ...] = REQUIRED_PAYLOAD_BYTES,
) -> tuple[AllReduceObservation, ...]:
    rows = []
    for payload in payloads:
        for repetition in range(2):
            for iteration in range(2):
                latency = 10.0 + payload * 0.002 + repetition + iteration / 10
                rows.append(
                    AllReduceObservation(
                        RAW_SCHEMA,
                        "fixture",
                        payload,
                        "float16",
                        2,
                        repetition,
                        iteration,
                        latency - 0.5,
                        latency,
                        latency,
                        20,
                        2,
                        "2026-09-07T00:00:00+00:00",
                    )
                )
    return tuple(rows)


def test_topology_capture_formats_command_once_without_mutating_input() -> None:
    capture = {
        "command": ["nvidia-smi", "topo", "-m"],
        "returncode": 0,
        "stdout": "GPU0 GPU1 PHB",
        "stderr": "",
    }
    original = {**capture, "command": list(capture["command"])}

    assert format_command_capture(capture) == (
        "COMMAND: nvidia-smi topo -m\n"
        "RETURN_CODE: 0\n"
        "STDOUT:\n"
        "GPU0 GPU1 PHB\n"
        "STDERR:\n"
    )
    assert capture == original


def write_raw(tmp_path: Path, rows: tuple[AllReduceObservation, ...]) -> tuple[Path, Path]:
    csv_path = tmp_path / "raw.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    json_path = tmp_path / "raw.json"
    json_path.write_text(
        json.dumps(
            {"schema_version": RAW_SCHEMA, "observations": [row.to_dict() for row in rows]}
        ),
        encoding="utf-8",
    )
    return csv_path, json_path


def write_m1_m2(tmp_path: Path, *, zero_throughput: bool = False) -> tuple[Path, Path]:
    m1 = tmp_path / "m1"
    m2 = tmp_path / "m2"
    m1.mkdir()
    m2.mkdir()
    for mode in ("graph", "eager"):
        (m1 / f"comparison-opt125m-{mode}-tp1-vs-tp2.json").write_text(
            json.dumps(
                {
                    "status": "comparison_complete",
                    "absolute": {
                        "baseline_output_tokens_per_second": 100.0,
                        "candidate_output_tokens_per_second": (
                            0.0 if zero_throughput else 80.0
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
    matrix = []
    for concurrency in (1, 4, 8, 16, 32, 64):
        matrix.extend(
            [
                {
                    "tensor_parallel_size": 1,
                    "concurrency": concurrency,
                    "output_throughput_tokens_per_second": 10.0 + concurrency,
                },
                {
                    "tensor_parallel_size": 2,
                    "concurrency": concurrency,
                    "output_throughput_tokens_per_second": 8.0 + 2 * concurrency,
                },
            ]
        )
    (m2 / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "kaggle-vllm-serving-summary-v1",
                "status": "executed",
                "matrix": matrix,
            }
        ),
        encoding="utf-8",
    )
    return m1, m2


def test_exact_linear_fit() -> None:
    summaries = [
        {"payload_bytes": size, "world_size": 2, "latency_us": {"mean": 7 + 0.002 * size}}
        for size in (1024, 4096, 16384, 65536)
    ]
    fit = fit_communication_model(summaries)
    assert fit["measured_allreduce_intercept_us"] == pytest.approx(7.0)
    assert fit["slope_us_per_byte"] == pytest.approx(0.002)
    assert fit["beta_effective_gb_s"] == pytest.approx(0.5)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert all(row["residual_us"] == pytest.approx(0.0) for row in fit["residuals"])


def test_fit_rejects_nonpositive_bandwidth_slope() -> None:
    summaries = [
        {"payload_bytes": size, "world_size": 2, "latency_us": {"mean": 10 - size / 10000}}
        for size in (1, 2, 3)
    ]
    with pytest.raises(CommunicationFitError, match="non-positive slope"):
        fit_communication_model(summaries)


def test_raw_csv_and_json_round_trip(tmp_path: Path) -> None:
    expected = observations()
    csv_path, json_path = write_raw(tmp_path, expected)
    csv_rows = load_allreduce_csv(csv_path)
    json_rows = load_allreduce_json(json_path)
    assert csv_rows == expected
    assert json_rows == expected
    assert_equivalent_ledgers(csv_rows, json_rows)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_latency_fails_closed(bad: float) -> None:
    rows = list(observations((1024, 4096)))
    rows[0] = replace(rows[0], rank1_latency_us=bad, critical_latency_us=bad)
    with pytest.raises(ResearchEvidenceError):
        validate_observations(rows, required_payloads=(1024, 4096))


@pytest.mark.parametrize("payload", [0, -1])
def test_nonpositive_payload_fails_closed(payload: int) -> None:
    rows = list(observations((1024, 4096)))
    rows[0] = replace(rows[0], payload_bytes=payload)
    with pytest.raises(ResearchEvidenceError):
        validate_observations(rows, required_payloads=(1024, 4096))


def test_critical_latency_requires_rank_maximum() -> None:
    rows = list(observations((1024, 4096)))
    rows[0] = replace(rows[0], critical_latency_us=rows[0].rank0_latency_us)
    path_row = rows[0].to_dict()
    with pytest.raises(ResearchEvidenceError, match="maximum rank latency"):
        from kaggle_vllm.research.measured_comm import _parse_observation

        _parse_observation(path_row)


def test_missing_payload_and_incomplete_grid_fail_closed() -> None:
    rows = observations((1024, 4096))
    with pytest.raises(ResearchEvidenceError, match="absent"):
        validate_observations(rows, required_payloads=(1024, 4096, 8192))
    with pytest.raises(ResearchEvidenceError, match="incomplete"):
        validate_observations(rows[:-1], required_payloads=(1024, 4096))


def test_malformed_csv_and_json_fail_closed(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("payload_bytes,latency\n1024,1\n", encoding="utf-8")
    json_path = tmp_path / "bad.json"
    json_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ResearchEvidenceError, match="columns"):
        load_allreduce_csv(csv_path)
    with pytest.raises(ResearchEvidenceError, match="parse"):
        load_allreduce_json(json_path)


def test_raw_ledgers_must_be_equal() -> None:
    rows = observations()
    changed = list(rows)
    changed[-1] = replace(changed[-1], captured_at_utc="different")
    with pytest.raises(ResearchEvidenceError, match="disagree"):
        assert_equivalent_ledgers(rows, changed)


def test_summary_reports_repetition_level_ci() -> None:
    summary = summarize_allreduce(observations((1024, 4096)))
    assert summary[0]["latency_us"]["count"] == 4
    assert summary[0]["latency_us"]["independent_count"] == 2
    assert summary[0]["latency_us"]["ci_unit"] == "independent_repetition_means"
    assert summary[0]["effective_payload_bandwidth_gb_s"] > 0
    assert summary[0]["effective_bandwidth_gb_s"]["count"] == 4
    assert summary[0]["effective_bandwidth_gb_s"]["mean_95_ci"][0] is not None


def test_architecture_requires_explicit_valid_metadata() -> None:
    with pytest.raises(ResearchEvidenceError, match="model_id"):
        parse_architecture_metadata({"architecture": "Unknown"})
    valid = parse_architecture_metadata(
        {
            "model_id": "caller/model",
            "revision": "abc",
            "architecture": "CallerVerifiedForCausalLM",
            "hidden_size": 2048,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "activation_dtype_bytes": 2,
            "source": "pinned config.json",
        }
    )
    assert derive_hidden_state_payload_bytes(valid, instantaneous_batch_tokens=8) == 32768


def test_comparison_never_substitutes_concurrency_for_batch(tmp_path: Path) -> None:
    m1, m2 = write_m1_m2(tmp_path)
    comparison = build_model_vs_observed(m1, m2, fit=None)
    assert len(comparison["rows"]) == 8
    assert all(row["payload_status"] == "unknown" for row in comparison["rows"])
    assert all(
        row["instantaneous_decode_batch"] == "unobserved"
        for row in comparison["rows"]
    )
    assert all(
        row["signed_service_time_difference_us"] is None
        for row in comparison["rows"]
    )


def test_comparison_rejects_zero_throughput(tmp_path: Path) -> None:
    m1, m2 = write_m1_m2(tmp_path, zero_throughput=True)
    with pytest.raises(ResearchEvidenceError, match="positive"):
        build_model_vs_observed(m1, m2, fit=None)


def test_comparison_rejects_incomplete_m2(tmp_path: Path) -> None:
    m1, m2 = write_m1_m2(tmp_path)
    payload = json.loads((m2 / "summary.json").read_text(encoding="utf-8"))
    payload["matrix"].pop()
    (m2 / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResearchEvidenceError, match="incomplete"):
        build_model_vs_observed(m1, m2, fit=None)


def test_comparison_rejects_incomplete_m1(tmp_path: Path) -> None:
    m1, m2 = write_m1_m2(tmp_path)
    (m1 / "comparison-opt125m-eager-tp1-vs-tp2.json").unlink()
    with pytest.raises(ResearchEvidenceError, match="parse"):
        build_model_vs_observed(m1, m2, fit=None)


def test_sha256_manifest_verification(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    item = evidence / "result.json"
    item.write_text("{}\n", encoding="utf-8")
    manifest = evidence / "SHA256SUMS.txt"
    manifest.write_text(f"{sha256_file(item)}  result.json\n", encoding="utf-8")
    assert verify_sha256_manifest(evidence, manifest) == {
        "result.json": sha256_file(item)
    }
    assert main(["verify-hashes", str(evidence)]) == 0
    item.write_text("changed", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="mismatch"):
        verify_sha256_manifest(evidence, manifest)


def test_sha256_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    manifest = tmp_path / "SHA256SUMS.txt"
    manifest.write_text(f"{'0' * 64}  ../outside\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="unsafe"):
        verify_sha256_manifest(tmp_path, manifest)


def test_report_generation(tmp_path: Path) -> None:
    m1, m2 = write_m1_m2(tmp_path)
    summaries = summarize_allreduce(observations())
    fit = fit_communication_model(summaries)
    comparison = build_model_vs_observed(m1, m2, fit=fit)
    output = tmp_path / "output"
    output.mkdir()
    written = write_m3_outputs(output, summaries=summaries, fit=fit, comparison=comparison)
    assert len(written) == 6
    assert all(path.is_file() for path in written)
    assert "not classical transport alpha" in (output / "M3_REPORT.md").read_text()
    assert (output / "M3_FIT_RESIDUALS.svg").is_file()


def test_cli_analyzes_equivalent_ledgers(tmp_path: Path) -> None:
    raw_csv, raw_json = write_raw(tmp_path, observations())
    m1, m2 = write_m1_m2(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    assert (
        main(
            [
                "analyze-m3",
                "--raw-csv",
                str(raw_csv),
                "--raw-json",
                str(raw_json),
                "--m1-dir",
                str(m1),
                "--m2-dir",
                str(m2),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "M3_FIT.json").is_file()


def test_lightweight_import_does_not_import_torch() -> None:
    command = [
        sys.executable,
        "-c",
        "import sys; import kaggle_vllm.research; assert 'torch' not in sys.modules",
    ]
    assert subprocess.run(command, check=False).returncode == 0


def test_disk_projection_fails_before_operation(tmp_path: Path) -> None:
    with pytest.raises(ResearchEvidenceError, match="exceeds limit"):
        require_disk_budget(
            tmp_path,
            projected_additional_bytes=1025,
            limit_bytes=1024,
        )


def test_nan_payload_summary_rejected() -> None:
    summaries = [
        {"payload_bytes": 1, "world_size": 2, "latency_us": {"mean": 1}},
        {"payload_bytes": 2, "world_size": 2, "latency_us": {"mean": math.nan}},
        {"payload_bytes": 3, "world_size": 2, "latency_us": {"mean": 3}},
    ]
    with pytest.raises(ResearchEvidenceError, match="finite"):
        fit_communication_model(summaries)
