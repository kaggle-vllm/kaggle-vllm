"""Online OpenAI-compatible serving concurrency benchmark and evidence schema."""

from __future__ import annotations

import concurrent.futures
import importlib.metadata
import json
import math
import re
import socket
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .benchmark import _safe_new_file, runtime_identity
from .environment import Environment, collect
from .exceptions import BenchmarkError
from .profiles import load_profile
from .telemetry import GPUMonitor, capture_topology

SERVING_SCHEMA = "kaggle-vllm-serving-benchmark-v1"
SUMMARY_SCHEMA = "kaggle-vllm-serving-summary-v1"
BENCHMARK_TYPE = "online-serving-concurrency"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
MILESTONE_CONCURRENCY = (1, 4, 8, 16, 32, 64)
VERIFIED_VLLM_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions",
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:request_success",
)
DEFAULT_PROMPTS = (
    (
        "Explain how tensor parallel inference partitions a transformer model, "
        "and distinguish compute throughput from memory capacity. Give a precise, "
        "self-contained answer with practical caveats."
    ),
    (
        "Describe how request concurrency affects prefill, decoding, KV-cache "
        "allocation, queueing, and time to first token in an online LLM server."
    ),
    (
        "Write a technical comparison of one-GPU and two-GPU inference on PCIe-"
        "connected accelerators. Avoid claiming that topology alone causes a result."
    ),
    (
        "Explain why an inference configuration can have lower single-request speed "
        "yet sustain a higher-capacity workload. Define the evidence needed to show it."
    ),
)

_CUDA_OOM = re.compile(
    r"(?:cuda\s+(?:error:\s*)?out\s+of\s+memory|"
    r"torch\.outofmemoryerror|cudacachingallocator.*out of memory)",
    re.IGNORECASE | re.DOTALL,
)
_PROM_SAMPLE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)(?:\s+\d+)?$"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class ServingWorkloadSpec:
    """Closed-loop streaming request workload for one concurrency point."""

    concurrency: int
    total_requests: int | None = None
    warmup_requests: int = 4
    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    prompt_profile: str = "fixed technical prompt corpus v1"
    max_output_tokens: int = 256
    temperature: float = 0.0
    ignore_eos: bool = True
    seed: int = 0
    request_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.concurrency, bool)
            or not isinstance(self.concurrency, int)
            or self.concurrency < 1
            or self.concurrency > 512
        ):
            raise ValueError("concurrency must be an integer in [1, 512]")
        resolved = self.total_requests
        if resolved is None:
            resolved = max(20, self.concurrency * 3)
            object.__setattr__(self, "total_requests", resolved)
        if (
            isinstance(resolved, bool)
            or not isinstance(resolved, int)
            or resolved < max(20, self.concurrency * 3)
        ):
            raise ValueError(
                "total_requests must provide at least 20 observations and three "
                "waves at the selected concurrency"
            )
        if (
            isinstance(self.warmup_requests, bool)
            or not isinstance(self.warmup_requests, int)
            or self.warmup_requests < 0
        ):
            raise ValueError("warmup_requests must be a non-negative integer")
        if not self.prompts or any(not value.strip() for value in self.prompts):
            raise ValueError("prompts must contain non-empty strings")
        if not self.prompt_profile.strip():
            raise ValueError("prompt_profile must be non-empty")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["prompts"] = list(self.prompts)
        data["dispatch_policy"] = "closed_loop_up_to_concurrency_in_flight"
        data["minimum_waves"] = 3
        return data


@dataclass(frozen=True)
class ServingBenchmarkSpec:
    """Model, server, and request settings declared for one online cell."""

    model: str
    tensor_parallel_size: int
    workload: ServingWorkloadSpec
    model_revision: str | None = None
    model_source: str = "huggingface"
    served_model_name: str = "qwen2.5-3b-instruct"
    base_url: str = "http://127.0.0.1:8000"
    dtype: str = "float16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = True
    disable_custom_all_reduce: bool = True
    max_num_batched_tokens: int | None = None
    max_num_seqs: int = 64
    telemetry_interval_seconds: float = 0.25

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.model_source not in {"huggingface", "local_transformers"}:
            raise ValueError("model_source must be 'huggingface' or 'local_transformers'")
        if self.model_source == "huggingface" and not self.model_revision:
            raise ValueError("a pinned model_revision is required for Hugging Face models")
        if self.tensor_parallel_size not in {1, 2}:
            raise ValueError("Milestone 2 tensor_parallel_size must be 1 or 2")
        if self.workload.concurrency > self.max_num_seqs:
            raise ValueError("max_num_seqs cannot be smaller than concurrency")
        if self.max_model_len < self.workload.max_output_tokens:
            raise ValueError("max_model_len cannot be smaller than max_output_tokens")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be positive when set")
        if self.telemetry_interval_seconds <= 0:
            raise ValueError("telemetry_interval_seconds must be positive")
        parsed = urlparse(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("base_url must be a notebook-local HTTP endpoint")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["workload"] = self.workload.to_dict()
        data["model_representation"] = "transformers"
        return data


@dataclass(frozen=True)
class RequestResult:
    """One measured streaming request, including failures without fake timings."""

    request_id: str
    concurrency: int
    start_timestamp_utc: str
    first_token_timestamp_utc: str | None
    completion_timestamp_utc: str
    input_tokens: int | None
    output_tokens: int | None
    ttft_seconds: float | None
    end_to_end_latency_seconds: float
    tpot_seconds: float | None
    status: str
    http_status: int | None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "completed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_failure(text: str, *, kind: str | None = None) -> str:
    """Classify only evidence present in an exception, response, or server log."""

    if _CUDA_OOM.search(text):
        return "CUDA_OOM_observed"
    normalized = (kind or "").casefold()
    if normalized in {"timeout", "timeouterror", "socket.timeout"}:
        return "client_timeout"
    if normalized in {"httperror", "http_error"}:
        return "HTTP_error"
    if normalized in {"urlerror", "connectionerror", "connection_error"}:
        return "connection_error"
    if normalized in {"server_exit", "server_start_failure"}:
        return normalized
    if text.strip():
        return "request_failure"
    return "unknown_failure"


def percentile(values: Sequence[float], percent: float) -> float | None:
    """Return a nearest-rank percentile (rank=ceil(p/100*n), one based)."""

    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100]")
    normalized = sorted(float(value) for value in values)
    if not normalized:
        return None
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("measurements must be finite")
    return normalized[max(0, math.ceil(percent / 100 * len(normalized)) - 1)]


def distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize a latency distribution; null means unavailable, never zero."""

    normalized = [float(value) for value in values]
    if normalized and not all(math.isfinite(value) for value in normalized):
        raise ValueError("measurements must be finite")
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized) if normalized else None,
        "median": statistics.median(normalized) if normalized else None,
        "p50": percentile(normalized, 50),
        "p95": percentile(normalized, 95),
        "p99": percentile(normalized, 99),
        "sample_standard_deviation": (
            statistics.stdev(normalized) if len(normalized) > 1 else None
        ),
        "min": min(normalized) if normalized else None,
        "max": max(normalized) if normalized else None,
    }


def calculate_tpot(
    first_token_elapsed: float | None,
    completion_elapsed: float,
    output_tokens: int | None,
) -> float | None:
    """Calculate post-first-token time per output token when mathematically defined."""

    if first_token_elapsed is None or output_tokens is None or output_tokens <= 1:
        return None
    return max(0.0, completion_elapsed - first_token_elapsed) / (output_tokens - 1)


def build_serving_plan(spec: ServingBenchmarkSpec, output: str | Path) -> dict[str, Any]:
    """Return a side-effect-free plan without runtime, GPU, model, or network access."""

    result_path = Path(output).expanduser()
    return {
        "schema_version": SERVING_SCHEMA,
        "benchmark_type": BENCHMARK_TYPE,
        "status": "planned_not_executed",
        "configuration": spec.to_dict(),
        "evidence": {
            "result": str(result_path),
            "requests": str(
                result_path.with_name(f"{result_path.stem}-requests.jsonl")
            ),
            "metrics": str(result_path.with_suffix(".metrics.txt")),
        },
        "metric_definitions": metric_definitions(),
        "operations": [
            "send excluded streaming warmup requests",
            "capture a Prometheus snapshot before measured dispatch",
            "sample every visible GPU across the measured interval",
            f"dispatch up to {spec.workload.concurrency} simultaneous HTTP requests",
            "capture a Prometheus snapshot after measured completion",
            "write new result, request JSONL, and raw metrics files",
        ],
        "mutations_performed": False,
    }


def metric_definitions() -> dict[str, str]:
    return {
        "ttft_seconds": "first content-bearing SSE event timestamp - request start",
        "tpot_seconds": (
            "(completion timestamp - first-token timestamp) / "
            "(actual output tokens - 1); null for <=1 token"
        ),
        "output_throughput_tokens_per_second": (
            "successful actual output tokens / measured wall-clock interval"
        ),
        "percentile_method": "nearest-rank: sorted[ceil(p/100*n)-1]",
    }


def _error_result(
    request_id: str,
    concurrency: int,
    started_at: datetime,
    start: float,
    clock: Callable[[], float],
    utc_now: Callable[[], datetime],
    error: BaseException,
    *,
    text: str = "",
) -> RequestResult:
    elapsed = max(0.0, clock() - start)
    kind = type(error).__name__
    evidence = text or str(error)
    if isinstance(error, (TimeoutError, socket.timeout)):
        kind = "timeout"
    elif isinstance(error, urllib.error.HTTPError):
        kind = "httperror"
    elif isinstance(error, urllib.error.URLError):
        kind = (
            "timeout"
            if isinstance(getattr(error, "reason", None), (TimeoutError, socket.timeout))
            else "urlerror"
        )
    return RequestResult(
        request_id=request_id,
        concurrency=concurrency,
        start_timestamp_utc=_iso(started_at),
        first_token_timestamp_utc=None,
        completion_timestamp_utc=_iso(utc_now()),
        input_tokens=None,
        output_tokens=None,
        ttft_seconds=None,
        end_to_end_latency_seconds=elapsed,
        tpot_seconds=None,
        status=classify_failure(evidence, kind=kind),
        http_status=getattr(error, "code", None),
        error_type=type(error).__name__,
        error_message=evidence[:4000] or kind,
    )


def perform_streaming_request(
    spec: ServingBenchmarkSpec,
    request_id: str,
    prompt: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    clock: Callable[[], float] = time.perf_counter,
    utc_now: Callable[[], datetime] = _utc_now,
) -> RequestResult:
    """Execute one genuine streaming chat-completion request."""

    payload = {
        "model": spec.served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": spec.workload.temperature,
        "max_completion_tokens": spec.workload.max_output_tokens,
        "ignore_eos": spec.workload.ignore_eos,
        "seed": spec.workload.seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{spec.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    started_at = utc_now()
    start = clock()
    first_elapsed: float | None = None
    first_at: datetime | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    http_status: int | None = None
    try:
        with opener(request, timeout=spec.workload.request_timeout_seconds) as response:
            http_status = int(getattr(response, "status", 200))
            if http_status != 200:
                body = response.read().decode("utf-8", errors="replace")
                raise urllib.error.HTTPError(
                    request.full_url, http_status, body, response.headers, None
                )
            for raw_line in response:
                if clock() - start > spec.workload.request_timeout_seconds:
                    raise TimeoutError("stream exceeded request timeout")
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data_text = line[5:].strip()
                if data_text == "[DONE]":
                    continue
                event = json.loads(data_text)
                usage = event.get("usage")
                if isinstance(usage, Mapping):
                    if usage.get("prompt_tokens") is not None:
                        prompt_tokens = int(usage["prompt_tokens"])
                    if usage.get("completion_tokens") is not None:
                        output_tokens = int(usage["completion_tokens"])
                choices = event.get("choices")
                if choices and isinstance(choices, list):
                    delta = choices[0].get("delta", {})
                    if (
                        isinstance(delta, Mapping)
                        and delta.get("content") is not None
                        and first_elapsed is None
                    ):
                        first_elapsed = max(0.0, clock() - start)
                        first_at = utc_now()
    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode("utf-8", errors="replace")
        except (AttributeError, OSError):
            body = str(error)
        return _error_result(
            request_id,
            spec.workload.concurrency,
            started_at,
            start,
            clock,
            utc_now,
            error,
            text=body,
        )
    except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        return _error_result(
            request_id,
            spec.workload.concurrency,
            started_at,
            start,
            clock,
            utc_now,
            error,
        )

    completion_elapsed = max(0.0, clock() - start)
    completed_at = utc_now()
    if first_elapsed is None or output_tokens is None or prompt_tokens is None:
        error = RuntimeError(
            "stream ended without a content event and complete server usage"
        )
        return RequestResult(
            request_id=request_id,
            concurrency=spec.workload.concurrency,
            start_timestamp_utc=_iso(started_at),
            first_token_timestamp_utc=_iso(first_at) if first_at else None,
            completion_timestamp_utc=_iso(completed_at),
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_seconds=first_elapsed,
            end_to_end_latency_seconds=completion_elapsed,
            tpot_seconds=None,
            status=classify_failure(str(error)),
            http_status=http_status,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    return RequestResult(
        request_id=request_id,
        concurrency=spec.workload.concurrency,
        start_timestamp_utc=_iso(started_at),
        first_token_timestamp_utc=_iso(first_at) if first_at else None,
        completion_timestamp_utc=_iso(completed_at),
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_seconds=first_elapsed,
        end_to_end_latency_seconds=completion_elapsed,
        tpot_seconds=calculate_tpot(first_elapsed, completion_elapsed, output_tokens),
        status="completed",
        http_status=http_status,
    )


def _run_requests(
    spec: ServingBenchmarkSpec,
    count: int,
    *,
    prefix: str,
    request_function: Callable[[ServingBenchmarkSpec, str, str], RequestResult],
    concurrency: int,
) -> list[RequestResult]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                request_function,
                spec,
                f"{prefix}-{index:04d}",
                spec.workload.prompts[index % len(spec.workload.prompts)],
            )
            for index in range(count)
        ]
        results = [future.result() for future in futures]
    return sorted(results, key=lambda item: item.request_id)


def parse_prometheus_metrics(text: str) -> dict[str, list[dict[str, Any]]]:
    """Parse only metric families verified in the pinned upstream vLLM source."""

    parsed = {name: [] for name in VERIFIED_VLLM_METRICS}
    for line in text.splitlines():
        match = _PROM_SAMPLE.match(line.strip())
        if not match or match.group("name") not in parsed:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        parsed[match.group("name")].append(
            {"labels": match.group("labels") or "", "value": value}
        )
    return parsed


def capture_metrics_snapshot(
    base_url: str,
    *,
    opener: Callable[..., Any] = urllib.request.urlopen,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Capture the actual endpoint, preserving absence rather than inventing names."""

    endpoint = f"{base_url.rstrip('/')}/metrics"
    try:
        with opener(endpoint, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
        if status != 200:
            return {"status": "HTTP_error", "http_status": status, "raw": raw, "parsed": {}}
        return {
            "status": "captured",
            "http_status": status,
            "raw": raw,
            "parsed": parse_prometheus_metrics(raw),
        }
    except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
        return {
            "status": "unavailable",
            "http_status": getattr(error, "code", None),
            "raw": "",
            "parsed": {},
            "error": str(error),
        }


def aggregate_request_results(
    requests: Sequence[RequestResult], measured_wall_seconds: float
) -> dict[str, Any]:
    """Aggregate successes while keeping every failure explicit."""

    if measured_wall_seconds <= 0:
        raise ValueError("measured_wall_seconds must be positive")
    successes = [request for request in requests if request.success]
    failures = [request for request in requests if not request.success]
    output_tokens = sum(request.output_tokens or 0 for request in successes)
    input_tokens = sum(request.input_tokens or 0 for request in successes)
    ttft = [request.ttft_seconds for request in successes if request.ttft_seconds is not None]
    tpot = [request.tpot_seconds for request in successes if request.tpot_seconds is not None]
    latency = [request.end_to_end_latency_seconds for request in successes]
    failure_counts: dict[str, int] = {}
    for request in failures:
        failure_counts[request.status] = failure_counts.get(request.status, 0) + 1
    return {
        "successful_requests": len(successes),
        "failed_requests": len(failures),
        "success_rate": len(successes) / len(requests) if requests else None,
        "failure_counts": dict(sorted(failure_counts.items())),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "measured_wall_seconds": measured_wall_seconds,
        "request_throughput_per_second": len(successes) / measured_wall_seconds,
        "input_throughput_tokens_per_second": input_tokens / measured_wall_seconds,
        "output_throughput_tokens_per_second": output_tokens / measured_wall_seconds,
        "ttft_seconds": distribution(ttft),
        "tpot_seconds": distribution(tpot),
        "latency_seconds": distribution(latency),
    }


def _metrics_file_text(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    sections = []
    for label, snapshot in (("BEFORE_MEASURED", before), ("AFTER_MEASURED", after)):
        sections.append(
            f"# KAGGLE_VLLM_SNAPSHOT {label}\n"
            f"# capture_status {snapshot.get('status')}\n"
            f"{snapshot.get('raw', '')}".rstrip()
        )
    return "\n\n".join(sections) + "\n"


def _write_text_new(path: Path, text: str) -> None:
    destination = _safe_new_file(path)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(text)


def run_serving_benchmark(
    spec: ServingBenchmarkSpec,
    output: str | Path,
    *,
    environment: Environment | None = None,
    request_function: Callable[
        [ServingBenchmarkSpec, str, str], RequestResult
    ] = perform_streaming_request,
    metrics_capture: Callable[[str], dict[str, Any]] = capture_metrics_snapshot,
    monitor_factory: Callable[[float], GPUMonitor] = GPUMonitor,
    topology_capture: Callable[[], dict[str, Any]] = capture_topology,
    clock: Callable[[], float] = time.perf_counter,
    utc_now: Callable[[], datetime] = _utc_now,
    server_log_path: Path | None = None,
    server_status: Callable[[], int | None] | None = None,
    server_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one online cell and write new, checksummable evidence files."""

    destination = _safe_new_file(output)
    requests_path = _safe_new_file(
        destination.with_name(f"{destination.stem}-requests.jsonl")
    )
    metrics_path = _safe_new_file(destination.with_suffix(".metrics.txt"))
    warmups = _run_requests(
        spec,
        spec.workload.warmup_requests,
        prefix="warmup",
        request_function=request_function,
        concurrency=min(spec.workload.concurrency, max(1, spec.workload.warmup_requests)),
    ) if spec.workload.warmup_requests else []
    before = metrics_capture(spec.base_url)
    measured_started_at = utc_now()
    measured_start = clock()
    with monitor_factory(spec.telemetry_interval_seconds) as monitor:
        requests = _run_requests(
            spec,
            int(spec.workload.total_requests or 0),
            prefix="request",
            request_function=request_function,
            concurrency=spec.workload.concurrency,
        )
    measured_wall = max(0.0, clock() - measured_start)
    measured_ended_at = utc_now()
    if measured_wall <= 0:
        raise BenchmarkError("measured request interval was not positive")
    after = metrics_capture(spec.base_url)

    runtime = environment or collect()
    identity, hardware = runtime_identity(runtime)
    identity["measurement_mode"] = "online_openai_chat_streaming"
    try:
        identity["vllm_distribution_version"] = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        pass
    profile = load_profile()
    identity["native_runtime"] = {
        "source_tag": profile.vllm_source_tag,
        "source_commit": profile.vllm_source_commit,
        "wheel": profile.wheel_filename,
        "sha256": profile.wheel_sha256,
        "hf_repository": profile.hf_repo_id,
        "hf_revision": profile.hf_revision,
    }
    aggregate = aggregate_request_results(requests, measured_wall)
    observed = sorted({request.status for request in requests if not request.success})
    log_oom = False
    if server_log_path is not None:
        try:
            log_oom = _CUDA_OOM.search(
                server_log_path.read_text(encoding="utf-8", errors="replace")
            ) is not None
        except OSError:
            pass
    if log_oom and "CUDA_OOM_observed" not in observed:
        observed.append("CUDA_OOM_observed")
    server_returncode = server_status() if server_status is not None else None
    if server_returncode is not None and "server_exit" not in observed:
        observed.append("server_exit")
    payload = {
        "schema_version": SERVING_SCHEMA,
        "benchmark_type": BENCHMARK_TYPE,
        "status": "executed",
        "identity": identity,
        "hardware": hardware,
        "topology": topology_capture(),
        "server": {
            "base_url": spec.base_url,
            "served_model_name": spec.served_model_name,
            "streaming": True,
            "api": "OpenAI-compatible chat completions",
            "unexpected_exit_returncode": server_returncode,
            **dict(server_metadata or {}),
        },
        "engine": {
            key: value
            for key, value in spec.to_dict().items()
            if key not in {"workload", "base_url"}
        },
        "workload": spec.workload.to_dict(),
        "concurrency": spec.workload.concurrency,
        "measurements": {
            **aggregate,
            "measured_started_at_utc": _iso(measured_started_at),
            "measured_ended_at_utc": _iso(measured_ended_at),
            "warmup_requests_excluded": len(warmups),
            "warmup_successful_requests": sum(item.success for item in warmups),
            "requests": {
                "storage": "jsonl",
                "file": requests_path.name,
                "count": len(requests),
            },
        },
        "gpu_telemetry": monitor.to_dict(),
        "metrics": {
            "endpoint": f"{spec.base_url.rstrip('/')}/metrics",
            "raw_file": metrics_path.name,
            "verified_against_vllm_source_commit": profile.vllm_source_commit,
            "verified_metric_names": list(VERIFIED_VLLM_METRICS),
            "before": {key: value for key, value in before.items() if key != "raw"},
            "after": {key: value for key, value in after.items() if key != "raw"},
        },
        "failure_observations": observed,
        "oom_observed": "CUDA_OOM_observed" in observed,
        "metric_definitions": metric_definitions(),
        "limitations": [
            "Sampled nvidia-smi telemetry can miss instantaneous memory peaks.",
            "Two T4 devices are separate GPUs, not one fungible 30 GB device.",
            "Client timestamps include loopback HTTP and scheduling effects.",
            "A throughput change alone is not evidence of CUDA OOM or KV-cache swapping.",
            "Results apply only to the recorded model, workload, engine, and runtime.",
        ],
    }
    validate_serving_result(payload)
    with requests_path.open("x", encoding="utf-8") as stream:
        for request in requests:
            stream.write(json.dumps(request.to_dict(), sort_keys=True) + "\n")
    _write_text_new(metrics_path, _metrics_file_text(before, after))
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def validate_serving_result(payload: Mapping[str, Any]) -> None:
    """Validate the stable Milestone 2 result contract."""

    if payload.get("schema_version") != SERVING_SCHEMA:
        raise BenchmarkError("unsupported serving benchmark schema_version")
    if payload.get("benchmark_type") != BENCHMARK_TYPE:
        raise BenchmarkError("unsupported serving benchmark type")
    if payload.get("status") not in {"executed", "failed"}:
        raise BenchmarkError("serving benchmark status must be executed or failed")
    for section in (
        "identity",
        "hardware",
        "topology",
        "server",
        "engine",
        "workload",
        "measurements",
        "gpu_telemetry",
        "metrics",
    ):
        if not isinstance(payload.get(section), Mapping):
            raise BenchmarkError(f"serving benchmark is missing {section!r}")
    if not isinstance(payload.get("failure_observations"), list):
        raise BenchmarkError("serving benchmark is missing failure_observations")
    measurements = payload["measurements"]
    for key in (
        "successful_requests",
        "failed_requests",
        "output_tokens",
        "measured_wall_seconds",
        "output_throughput_tokens_per_second",
        "ttft_seconds",
        "tpot_seconds",
        "latency_seconds",
    ):
        if key not in measurements:
            raise BenchmarkError(f"serving measurements are missing {key!r}")
    for name in ("ttft_seconds", "tpot_seconds", "latency_seconds"):
        item = measurements[name]
        if not isinstance(item, Mapping) or "p95" not in item:
            raise BenchmarkError(f"serving distribution {name!r} is invalid")


def load_serving_result(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"unable to read serving result {path}: {error}") from error
    if not isinstance(payload, dict):
        raise BenchmarkError("serving benchmark result must be a JSON object")
    validate_serving_result(payload)
    return payload


def analyze_crossover(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Find throughput and capacity crossovers without assuming either exists."""

    indexed = {
        (int(cell["engine"]["tensor_parallel_size"]), int(cell["concurrency"])): cell
        for cell in cells
    }

    def first(predicate: Callable[[Mapping[str, Any]], bool], tp: int) -> int | None:
        for concurrency in MILESTONE_CONCURRENCY:
            cell = indexed.get((tp, concurrency))
            if cell is not None and predicate(cell):
                return concurrency
        return None

    throughput_crossover = None
    capacity_crossover = None
    for concurrency in MILESTONE_CONCURRENCY:
        tp1 = indexed.get((1, concurrency))
        tp2 = indexed.get((2, concurrency))
        if tp1 is None or tp2 is None:
            continue
        m1, m2 = tp1["measurements"], tp2["measurements"]
        t1 = m1.get("output_throughput_tokens_per_second")
        t2 = m2.get("output_throughput_tokens_per_second")
        r1, r2 = m1.get("success_rate"), m2.get("success_rate")
        if (
            throughput_crossover is None
            and t1 is not None
            and t2 is not None
            and r1 is not None
            and r2 is not None
            and float(t2) > float(t1)
            and float(r2) >= float(r1)
        ):
            throughput_crossover = concurrency
        if (
            capacity_crossover is None
            and int(m1.get("failed_requests", 0)) > 0
            and int(m2.get("failed_requests", 0)) == 0
        ):
            capacity_crossover = concurrency

    def failed(cell: Mapping[str, Any]) -> bool:
        return int(cell["measurements"].get("failed_requests", 0)) > 0

    def oom(cell: Mapping[str, Any]) -> bool:
        return bool(cell.get("oom_observed"))

    return {
        "throughput_crossover_concurrency": throughput_crossover,
        "throughput_crossover_reason": (
            "First tested concurrency where TP2 throughput exceeded TP1 and TP2 "
            "maintained an equal or higher request success rate."
            if throughput_crossover is not None
            else "No TP2 throughput crossover observed in tested range."
        ),
        "capacity_crossover_concurrency": capacity_crossover,
        "capacity_crossover_reason": (
            "First tested concurrency where TP1 had request failures and TP2 completed "
            "all requests."
            if capacity_crossover is not None
            else "No TP1-fails/TP2-survives capacity crossover observed in tested range."
        ),
        "tp1_first_failure_concurrency": first(failed, 1),
        "tp2_first_failure_concurrency": first(failed, 2),
        "tp1_first_cuda_oom_concurrency": first(oom, 1),
        "tp2_first_cuda_oom_concurrency": first(oom, 2),
    }


def summary_row(cell: Mapping[str, Any]) -> dict[str, Any]:
    measurements = cell["measurements"]
    telemetry = cell["gpu_telemetry"]
    return {
        "tensor_parallel_size": cell["engine"]["tensor_parallel_size"],
        "concurrency": cell["concurrency"],
        "output_throughput_tokens_per_second": measurements.get(
            "output_throughput_tokens_per_second"
        ),
        "ttft_p95_seconds": measurements["ttft_seconds"].get("p95"),
        "tpot_p95_seconds": measurements["tpot_seconds"].get("p95"),
        "successful_requests": measurements.get("successful_requests"),
        "failed_requests": measurements.get("failed_requests"),
        "cuda_oom_observed": bool(cell.get("oom_observed")),
        "per_gpu_peak_sampled_memory_used_mib": [
            {"index": item.get("index"), "peak_memory_used_mib": item.get("peak_memory_used_mib")}
            for item in telemetry.get("summaries", [])
        ],
        "maximum_aggregate_sampled_memory_used_mib": telemetry.get(
            "maximum_aggregate_sampled_memory_used_mib"
        ),
    }
