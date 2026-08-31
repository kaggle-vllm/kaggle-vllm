from __future__ import annotations

from kaggle_vllm.telemetry import (
    CommandCapture,
    GPUMonitor,
    capture_topology,
    parse_telemetry_csv,
    parse_topology_matrix,
    run_command,
    summarize_gpu_samples,
)

TOPOLOGY = """\
GPU0\tGPU1\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID
GPU0\t X \tPHB\t0-3\t0\tN/A
GPU1\tPHB\t X \t0-3\t0\tN/A

Legend:
  X = Self
  PHB = Connection traversing a PCIe Host Bridge
  NV# = Connection traversing bonded NVLinks
"""


def test_parse_topology_records_observed_phb_without_nvlink():
    parsed = parse_topology_matrix(f"\x1b[4m{TOPOLOGY}\x1b[0m")
    assert parsed == {
        "gpu_labels": ["GPU0", "GPU1"],
        "links": [{"gpu_a": "GPU0", "gpu_b": "GPU1", "path": "PHB"}],
        "nvlink_observed": False,
    }


def test_parse_topology_unknown_when_matrix_is_unavailable():
    assert parse_topology_matrix("not a topology table") == {
        "gpu_labels": [],
        "links": [],
        "nvlink_observed": None,
    }


def test_capture_topology_preserves_command_failures():
    def runner(command):
        if tuple(command)[-1] == "-m":
            return CommandCapture(tuple(command), "ok", 0, TOPOLOGY, "")
        return CommandCapture(tuple(command), "error", 1, "", "unsupported")

    captured = capture_topology(runner=runner)
    assert captured["parsed_matrix"]["links"][0]["path"] == "PHB"
    assert captured["peer_access_read"]["status"] == "error"
    assert captured["peer_access_write"]["stderr"] == "unsupported"


def test_parse_and_summarize_gpu_telemetry_with_unsupported_values():
    text = "0, 512, 80, 45, 51.5, 1200, 5000\n1, 256, N/A, 44, 40, 1100, 4900"
    samples = parse_telemetry_csv(text, captured_at="2026-08-31T00:00:00+00:00")
    assert len(samples) == 2
    assert samples[1].utilization_percent is None
    summaries = summarize_gpu_samples(samples)
    assert summaries[0]["peak_memory_used_mib"] == 512
    assert summaries[0]["mean_utilization_percent"] == 80
    assert summaries[1]["peak_utilization_percent"] is None


def test_monitor_collects_initial_and_final_samples_without_cuda():
    def runner(command):
        return CommandCapture(
            tuple(command), "ok", 0, "0, 10, 20, 30, 40, 50, 60", ""
        )

    with GPUMonitor(interval_seconds=60, runner=runner) as monitor:
        pass
    payload = monitor.to_dict()
    assert payload["summaries"][0]["sample_count"] == 2
    assert payload["capture_failures"] == []
    assert payload["raw_samples_retained"] is False


def test_run_command_reports_missing_executable():
    result = run_command(("definitely-not-a-real-kaggle-vllm-command",))
    assert result.status == "unavailable"
    assert result.returncode is None
