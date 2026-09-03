from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from kaggle_vllm.cli import main
from kaggle_vllm.environment import Environment, GPUInfo
from kaggle_vllm.exceptions import BenchmarkError
from kaggle_vllm.serving_benchmark import (
    BENCHMARK_TYPE,
    DEFAULT_MODEL_REVISION,
    SERVING_SCHEMA,
    RequestResult,
    ServingBenchmarkSpec,
    ServingWorkloadSpec,
    aggregate_request_results,
    analyze_crossover,
    build_serving_plan,
    calculate_tpot,
    classify_failure,
    distribution,
    parse_prometheus_metrics,
    percentile,
    perform_streaming_request,
    run_serving_benchmark,
    validate_serving_result,
)


def spec(**options):
    workload_options = options.pop("workload_options", {})
    concurrency = workload_options.pop("concurrency", 1)
    return ServingBenchmarkSpec(
        model="Qwen/Qwen2.5-3B-Instruct",
        model_revision=DEFAULT_MODEL_REVISION,
        tensor_parallel_size=options.pop("tensor_parallel_size", 1),
        workload=ServingWorkloadSpec(concurrency=concurrency, **workload_options),
        **options,
    )


@pytest.mark.parametrize("value", [0, -1, 1.5, True, 513])
def test_invalid_concurrency_is_rejected(value):
    with pytest.raises(ValueError, match="concurrency"):
        ServingWorkloadSpec(concurrency=value)


def test_request_count_requires_meaningful_percentile_sample():
    with pytest.raises(ValueError, match="three waves"):
        ServingWorkloadSpec(concurrency=8, total_requests=23)
    workload = ServingWorkloadSpec(concurrency=8, total_requests=24)
    assert workload.total_requests == 24
    assert workload.warmup_requests == 8
    assert ServingWorkloadSpec(concurrency=1).total_requests == 20


def test_percentile_uses_documented_nearest_rank_method():
    values = list(range(1, 21))
    assert percentile(values, 95) == 19
    assert distribution(values)["p95"] == 19
    assert distribution([])["p95"] is None
    with pytest.raises(ValueError, match="percent"):
        percentile(values, 0)


def test_tpot_formula_and_short_output_null():
    assert calculate_tpot(0.5, 2.5, 5) == pytest.approx(0.5)
    assert calculate_tpot(0.5, 2.5, 1) is None
    assert calculate_tpot(0.5, 2.5, 0) is None
    assert calculate_tpot(None, 2.5, 5) is None


class FakeResponse:
    def __init__(self, lines, status=200, body=b""):
        self.lines = lines
        self.status = status
        self.body = body
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return self.body


class StepClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class WallClock:
    def __init__(self):
        self.value = datetime(2026, 9, 2, tzinfo=timezone.utc)

    def __call__(self):
        result = self.value
        self.value += timedelta(milliseconds=100)
        return result


def test_streaming_request_measures_first_content_event_and_server_usage():
    lines = [
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3}}\n',
        b"data: [DONE]\n",
    ]
    clock = StepClock((0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2))
    result = perform_streaming_request(
        spec(workload_options={"total_requests": 20, "warmup_requests": 0}),
        "request-0000",
        "prompt",
        opener=lambda *_args, **_kwargs: FakeResponse(lines),
        clock=clock,
        utc_now=WallClock(),
    )
    assert result.status == "completed"
    assert result.ttft_seconds == pytest.approx(0.5)
    assert result.output_tokens == 3
    assert result.input_tokens == 12
    assert result.tpot_seconds == pytest.approx(0.35)


def request_result(*, status="completed", output_tokens=4):
    return RequestResult(
        request_id="r",
        concurrency=2,
        start_timestamp_utc="2026-09-02T00:00:00+00:00",
        first_token_timestamp_utc=(
            "2026-09-02T00:00:00.1+00:00" if status == "completed" else None
        ),
        completion_timestamp_utc="2026-09-02T00:00:01+00:00",
        input_tokens=3 if status == "completed" else None,
        output_tokens=output_tokens if status == "completed" else None,
        ttft_seconds=0.1 if status == "completed" else None,
        end_to_end_latency_seconds=1.0,
        tpot_seconds=(0.3 if status == "completed" and output_tokens > 1 else None),
        status=status,
        http_status=200 if status == "completed" else None,
    )


def test_aggregate_uses_successful_actual_tokens_and_preserves_failures():
    aggregate = aggregate_request_results(
        [request_result(), request_result(status="client_timeout")], 2.0
    )
    assert aggregate["successful_requests"] == 1
    assert aggregate["failed_requests"] == 1
    assert aggregate["output_tokens"] == 4
    assert aggregate["output_throughput_tokens_per_second"] == 2
    assert aggregate["ttft_seconds"]["count"] == 1
    assert aggregate["failure_counts"] == {"client_timeout": 1}


def test_failure_classifier_requires_real_oom_text():
    assert classify_failure("CUDA out of memory. Tried to allocate") == "CUDA_OOM_observed"
    assert classify_failure("slow throughput") == "request_failure"
    assert classify_failure("", kind="timeout") == "client_timeout"
    assert classify_failure("", kind="other") == "unknown_failure"


def test_metrics_parser_keeps_only_verified_names_and_handles_missing():
    parsed = parse_prometheus_metrics(
        'vllm:num_requests_running{model_name="qwen"} 2\n'
        "vllm:kv_cache_usage_perc 0.75\n"
        "invented_metric 9\n"
    )
    assert parsed["vllm:num_requests_running"][0]["value"] == 2
    assert parsed["vllm:kv_cache_usage_perc"][0]["value"] == 0.75
    assert parsed["vllm:num_requests_waiting"] == []
    assert "invented_metric" not in parsed


def test_dry_run_plan_is_side_effect_free(tmp_path, monkeypatch, capsys):
    output = tmp_path / "result.json"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run attempted a runtime side effect")

    monkeypatch.setattr("kaggle_vllm.environment.collect", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    assert (
        main(
            [
                "benchmark-serving",
                "--model",
                "Qwen/Qwen2.5-3B-Instruct",
                "--tensor-parallel-size",
                "1",
                "--concurrency",
                "16",
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "planned_not_executed"
    assert payload["configuration"]["workload"]["total_requests"] == 48
    assert payload["configuration"]["workload"]["warmup_requests"] == 16
    assert not output.exists()
    assert not (tmp_path / "result-requests.jsonl").exists()
    assert not (tmp_path / "result.metrics.txt").exists()
    assert not (tmp_path / "result.telemetry.jsonl").exists()


def fake_environment():
    return Environment(
        is_kaggle=True,
        python="3.12.13",
        platform="Linux-test",
        torch="2.10.0+cu128",
        torch_path="/system/torch/__init__.py",
        torch_cuda="12.8",
        cuda_available=True,
        gpus=(GPUInfo(0, "Tesla T4", (7, 5), 15_636_037_632),),
        nccl="2.27.5",
        nvcc="nvcc",
        nvcc_version="12.8",
        cuda_home="/usr/local/cuda",
        cuda_driver="libcuda.so",
        cmake_library_path=None,
    )


class FakeMonitor:
    def __init__(self, _interval):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def to_dict(self):
        return {
            "telemetry_sample_count": 2,
            "maximum_aggregate_sampled_memory_used_mib": 100,
            "summaries": [{"index": 0, "peak_memory_used_mib": 100}],
        }


def test_schema_serialization_validation_and_safe_new_outputs(tmp_path):
    output = tmp_path / "result.json"
    counter = iter((0.0, 2.0))

    def fake_request(_specification, request_id, _prompt):
        return RequestResult(**{**request_result().to_dict(), "request_id": request_id})

    payload = run_serving_benchmark(
        spec(workload_options={"total_requests": 20, "warmup_requests": 0}),
        output,
        environment=fake_environment(),
        request_function=fake_request,
        metrics_capture=lambda _url: {
            "status": "captured",
            "http_status": 200,
            "raw": "vllm:num_requests_running 0\n",
            "parsed": {"vllm:num_requests_running": [{"labels": "", "value": 0}]},
        },
        monitor_factory=FakeMonitor,
        topology_capture=lambda: {"parsed_matrix": {"links": []}},
        clock=lambda: next(counter),
    )
    assert payload["schema_version"] == SERVING_SCHEMA
    assert payload["benchmark_type"] == BENCHMARK_TYPE
    assert payload["measurements"]["output_tokens"] == 80
    assert output.is_file()
    assert (tmp_path / "result-requests.jsonl").is_file()
    assert (tmp_path / "result.metrics.txt").is_file()
    assert (tmp_path / "result.telemetry.jsonl").is_file()
    assert payload["measurements"]["input_tokens_per_request"]["count"] == 20
    validate_serving_result(json.loads(output.read_text()))
    with pytest.raises(BenchmarkError, match="overwrite"):
        run_serving_benchmark(
            spec(workload_options={"total_requests": 20, "warmup_requests": 0}),
            output,
        )
    with pytest.raises(BenchmarkError, match="schema_version"):
        validate_serving_result({"schema_version": "wrong"})


def test_serving_output_rejects_symlink_traversal_before_requests(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(BenchmarkError, match="symlink"):
        run_serving_benchmark(
            spec(workload_options={"total_requests": 20, "warmup_requests": 0}),
            link / "result.json",
        )


def crossover_cell(tp, concurrency, throughput, failures=0, oom=False):
    successes = 20 - failures
    return {
        "engine": {"tensor_parallel_size": tp},
        "concurrency": concurrency,
        "measurements": {
            "output_throughput_tokens_per_second": throughput,
            "success_rate": successes / 20,
            "successful_requests": successes,
            "failed_requests": failures,
        },
        "oom_observed": oom,
    }


def test_crossover_detection_normal_and_no_crossover():
    cells = []
    for concurrency in (1, 4, 8, 16, 32, 64):
        cells.extend(
            [
                crossover_cell(1, concurrency, 100),
                crossover_cell(2, concurrency, 110 if concurrency >= 16 else 90),
            ]
        )
    assert analyze_crossover(cells)["throughput_crossover_concurrency"] == 16
    for cell in cells:
        if cell["engine"]["tensor_parallel_size"] == 2:
            cell["measurements"]["output_throughput_tokens_per_second"] = 80
    result = analyze_crossover(cells)
    assert result["throughput_crossover_concurrency"] is None
    assert "No TP2" in result["throughput_crossover_reason"]


def test_crossover_detection_capacity_throughput_and_oom_are_independent():
    cells = [
        crossover_cell(1, 32, 50, failures=2, oom=True),
        crossover_cell(2, 32, 45),
        crossover_cell(1, 64, 40, failures=0),
        crossover_cell(2, 64, 60, failures=1),
    ]
    result = analyze_crossover(cells)
    assert result["capacity_crossover_concurrency"] == 32
    assert result["throughput_crossover_concurrency"] is None
    assert result["tp1_first_failure_concurrency"] == 32
    assert result["tp2_first_failure_concurrency"] == 64
    assert result["tp1_first_cuda_oom_concurrency"] == 32


def test_plan_records_real_concurrency_not_offline_batching(tmp_path):
    plan = build_serving_plan(
        spec(workload_options={"concurrency": 4, "total_requests": 20}),
        tmp_path / "result.json",
    )
    assert plan["configuration"]["workload"]["dispatch_policy"].startswith(
        "closed_loop"
    )
    assert plan["metric_definitions"]["ttft_seconds"].startswith("first content")
