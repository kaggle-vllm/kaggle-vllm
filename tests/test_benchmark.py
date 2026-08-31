from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kaggle_vllm.benchmark import (
    BenchmarkSpec,
    WorkloadSpec,
    aggregate_trials,
    build_benchmark_plan,
    prepare_evidence_directory,
    run_offline_benchmark,
    summarize,
    write_json_new,
)
from kaggle_vllm.benchmark_compare import (
    compare_results,
    load_benchmark_result,
    percentage_delta,
    validate_benchmark_result,
)
from kaggle_vllm.environment import Environment, GPUInfo
from kaggle_vllm.exceptions import BenchmarkError, RuntimeValidationError


def fake_environment(gpu_count=2):
    return Environment(
        is_kaggle=True,
        python="3.12.13",
        platform="Linux-test-x86_64-with-glibc2.35",
        torch="2.10.0+cu128",
        torch_path="/system/torch/__init__.py",
        torch_cuda="12.8",
        cuda_available=True,
        gpus=tuple(
            GPUInfo(index, "Tesla T4", (7, 5), 15_636_037_632)
            for index in range(gpu_count)
        ),
        nccl="2.27.5",
        nvcc="/usr/local/cuda/bin/nvcc",
        nvcc_version="Cuda compilation tools, release 12.8, V12.8.93",
        cuda_home="/usr/local/cuda",
        cuda_driver="/usr/local/nvidia/lib64/libcuda.so",
        cmake_library_path=None,
        driver_version="580.159.04",
        driver_reported_cuda_max="13.0",
    )


def test_benchmark_spec_round_trip_and_sharded_load_format():
    spec = BenchmarkSpec(
        model="/model",
        model_revision="abc",
        model_representation="sharded_state",
        tensor_parallel_size=2,
        max_model_len=2048,
        max_num_batched_tokens=4096,
    )
    assert spec.effective_load_format == "sharded_state"
    assert BenchmarkSpec.from_mapping(spec.to_dict()) == spec


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tensor_parallel_size": 0}, "tensor_parallel_size"),
        ({"gpu_memory_utilization": 0}, "gpu_memory_utilization"),
        ({"max_num_batched_tokens": 0}, "max_num_batched_tokens"),
        ({"max_num_seqs": 2}, "request count"),
    ],
)
def test_benchmark_spec_validation(kwargs, message):
    options = {"model": "model", "tensor_parallel_size": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        BenchmarkSpec(**options)


def test_workload_validation_and_statistics():
    with pytest.raises(ValueError, match="measurement_runs"):
        WorkloadSpec(measurement_runs=0)
    assert summarize([1, 2, 3]) == {
        "count": 3,
        "mean": 2.0,
        "median": 2.0,
        "sample_standard_deviation": 1.0,
        "min": 1.0,
        "max": 3.0,
    }
    assert summarize([5])["sample_standard_deviation"] is None


def test_percentage_delta_handles_direction_and_zero_denominator():
    assert percentage_delta(100.0, 75.0) == -25.0
    assert percentage_delta(100.0, 125.0) == 25.0
    assert percentage_delta(0.0, 1.0) is None
    assert percentage_delta(None, 1.0) is None


def test_aggregate_trials_reports_variation_and_exact_totals():
    trials = [
        {
            "wall_duration_seconds": 1.0,
            "aggregate_input_tokens_per_second": 8.0,
            "aggregate_output_tokens_per_second": 16.0,
            "request_completion_wall_seconds": [1.0, 1.0],
            "input_tokens": 8,
            "output_tokens": 16,
            "requests_completed": 2,
        },
        {
            "wall_duration_seconds": 2.0,
            "aggregate_input_tokens_per_second": 4.0,
            "aggregate_output_tokens_per_second": 8.0,
            "request_completion_wall_seconds": [2.0, 2.0],
            "input_tokens": 8,
            "output_tokens": 16,
            "requests_completed": 2,
        },
    ]
    aggregate = aggregate_trials(trials)
    assert aggregate["requests_completed"] == 4
    assert aggregate["output_tokens"] == 32
    assert aggregate["aggregate_output_tokens_per_second"] == pytest.approx(32 / 3)
    assert aggregate["trial_wall_duration_seconds"]["median"] == 1.5


def test_dry_run_plan_has_no_filesystem_or_gpu_side_effect(tmp_path):
    output = tmp_path / "not-created.json"
    plan = build_benchmark_plan(
        BenchmarkSpec(model="model", tensor_parallel_size=2), output
    )
    assert plan["status"] == "planned_not_executed"
    assert plan["engine"]["tensor_parallel_size"] == 2
    assert not output.exists()


def test_evidence_output_rejects_existing_and_symlink_paths(tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="overwrite"):
        write_json_new(existing, {})

    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(BenchmarkError, match="symlink"):
        write_json_new(link / "evidence.json", {})


def test_prepare_evidence_directory_requires_absent_target(tmp_path):
    destination = prepare_evidence_directory(tmp_path / "bundle")
    assert destination.is_dir()
    with pytest.raises(BenchmarkError, match="reuse"):
        prepare_evidence_directory(destination)
    with pytest.raises(BenchmarkError, match="/kaggle/input"):
        prepare_evidence_directory("/kaggle/input/forbidden-evidence")


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    options = None

    def __init__(self, **kwargs):
        FakeLLM.options = kwargs

    def generate(self, prompts, _sampling):
        return [
            SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[SimpleNamespace(token_ids=[4, 5, 6, 7])],
            )
            for _ in prompts
        ]


class FakeMonitor:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def to_dict(self):
        return {
            "source": "fixture",
            "summaries": [
                {"index": 0, "peak_memory_used_mib": 100, "peak_utilization_percent": 80},
                {"index": 1, "peak_memory_used_mib": 90, "peak_utilization_percent": 70},
            ],
        }


def test_run_offline_benchmark_with_injected_cpu_fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "kaggle_vllm.benchmark._load_vllm", lambda: (FakeLLM, FakeSamplingParams)
    )
    times = iter((0.0, 2.0, 3.0, 4.0, 5.0, 7.0))
    spec = BenchmarkSpec(
        model="model",
        model_revision="immutable",
        tensor_parallel_size=2,
        max_num_batched_tokens=4096,
        workload=WorkloadSpec(
            prompts=("one", "two"), warmup_runs=1, measurement_runs=2
        ),
    )
    output = tmp_path / "result.json"
    payload = run_offline_benchmark(
        spec,
        output,
        environment=fake_environment(),
        clock=lambda: next(times),
        monitor_factory=FakeMonitor,
        topology_capture=lambda: {
            "parsed_matrix": {
                "links": [{"gpu_a": "GPU0", "gpu_b": "GPU1", "path": "PHB"}],
                "nvlink_observed": False,
            }
        },
    )
    assert output.is_file()
    assert payload["measurements"]["model_load_wall_seconds"] == 2
    assert payload["measurements"]["aggregate"]["requests_completed"] == 4
    assert payload["measurements"]["aggregate"]["output_tokens"] == 16
    assert payload["engine"]["max_num_batched_tokens"] == 4096
    assert FakeLLM.options["tensor_parallel_size"] == 2
    assert FakeLLM.options["revision"] == "immutable"
    assert load_benchmark_result(output)["status"] == "executed"


def test_run_offline_benchmark_checks_tp_against_visible_gpus(tmp_path):
    with pytest.raises(RuntimeValidationError, match="found 1"):
        run_offline_benchmark(
            BenchmarkSpec(model="model", tensor_parallel_size=2),
            tmp_path / "result.json",
            environment=fake_environment(gpu_count=1),
        )


def result_fixture(*, tp, throughput, latency, memory, max_batched=None):
    return {
        "schema_version": 1,
        "status": "executed",
        "identity": {"measurement_mode": "offline_llm_generate"},
        "hardware": {},
        "topology": {
            "parsed_matrix": {
                "links": [{"gpu_a": "GPU0", "gpu_b": "GPU1", "path": "PHB"}],
                "nvlink_observed": False,
            }
        },
        "engine": {
            "model": "model",
            "model_revision": "rev",
            "model_representation": "transformers",
            "load_format": None,
            "dtype": "float16",
            "max_model_len": 512,
            "gpu_memory_utilization": 0.4,
            "enforce_eager": False,
            "disable_custom_all_reduce": True,
            "max_num_batched_tokens": max_batched,
            "max_num_seqs": None,
            "tensor_parallel_size": tp,
        },
        "workload": {"prompts": ["same"], "measurement_runs": 3},
        "measurements": {
            "model_load_wall_seconds": 10,
            "aggregate": {
                "aggregate_input_tokens_per_second": 10,
                "aggregate_output_tokens_per_second": throughput,
                "total_wall_duration_seconds": 3,
                "requests_completed": 3,
                "trial_wall_duration_seconds": {"mean": latency},
            },
        },
        "gpu_telemetry": {
            "summaries": [
                {
                    "index": index,
                    "peak_memory_used_mib": memory,
                    "peak_utilization_percent": 50,
                }
                for index in range(tp)
            ]
        },
        "limitations": [],
    }


def test_compare_results_reports_neutral_controlled_tp_deltas():
    comparison = compare_results(
        result_fixture(tp=1, throughput=100, latency=1, memory=1000),
        result_fixture(tp=2, throughput=75, latency=1.5, memory=600),
        baseline_label="tp1",
        candidate_label="tp2",
    )
    assert comparison["comparability"]["controlled_tp_only"] is True
    assert comparison["deltas"]["output_throughput_percent"] == -25
    assert comparison["deltas"]["mean_batch_latency_percent"] == 50
    assert comparison["deltas"]["mean_peak_memory_per_device_mib"] == -400
    assert any("no NVLink" in value for value in comparison["observations"])
    assert any("not causal" in value for value in comparison["limitations"])


def test_compare_results_marks_batching_change_as_not_tp_only():
    comparison = compare_results(
        result_fixture(tp=2, throughput=100, latency=1, memory=600),
        result_fixture(
            tp=2, throughput=110, latency=0.9, memory=620, max_batched=4096
        ),
    )
    assert comparison["comparability"]["controlled_tp_only"] is False
    assert comparison["comparability"]["differing_non_tp_engine_fields"] == [
        "max_num_batched_tokens"
    ]


def test_result_validation_rejects_historical_or_incomplete_schema(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"schema_version": 1, "status": "executed"}))
    with pytest.raises(BenchmarkError, match="missing"):
        load_benchmark_result(path)
    with pytest.raises(BenchmarkError, match="schema_version"):
        validate_benchmark_result({"schema_version": 2})
