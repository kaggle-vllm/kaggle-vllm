"""Reproducible offline vLLM benchmark evidence and neutral comparisons."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bootstrap import activate_runtime
from .environment import Environment, collect
from .exceptions import BenchmarkError, VLLMNotInstalledError
from .profiles import load_profile
from .runtime import validate_tensor_parallel_size
from .sharding import inspect_sharded_model
from .telemetry import GPUMonitor, capture_topology

SCHEMA_VERSION = 1
MEASUREMENT_MODE = "offline_llm_generate"
DEFAULT_PROMPTS = (
    "Explain tensor parallel inference in one paragraph.",
    "Describe one reason to use two GPUs for model inference.",
    "What role does NCCL play in distributed GPU applications?",
    "Why can a small model run slower with tensor parallelism?",
)


@dataclass(frozen=True)
class WorkloadSpec:
    """Deterministic prompts and repetition policy for one benchmark run."""

    prompts: tuple[str, ...] = DEFAULT_PROMPTS
    max_output_tokens: int = 128
    warmup_runs: int = 1
    measurement_runs: int = 5
    temperature: float = 0.0
    ignore_eos: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.prompts or any(not prompt.strip() for prompt in self.prompts):
            raise ValueError("workload prompts must contain non-empty strings")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if self.measurement_runs < 1:
            raise ValueError("measurement_runs must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be a finite non-negative number")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> WorkloadSpec:
        return cls(
            prompts=tuple(str(value) for value in data.get("prompts", DEFAULT_PROMPTS)),
            max_output_tokens=int(data.get("max_output_tokens", 128)),
            warmup_runs=int(data.get("warmup_runs", 1)),
            measurement_runs=int(data.get("measurement_runs", 5)),
            temperature=float(data.get("temperature", 0.0)),
            ignore_eos=bool(data.get("ignore_eos", True)),
            seed=int(data.get("seed", 0)),
        )


@dataclass(frozen=True)
class BenchmarkSpec:
    """Validated model, engine, and workload settings for one offline run."""

    model: str
    tensor_parallel_size: int
    model_revision: str | None = None
    model_representation: str = "transformers"
    load_format: str | None = None
    dtype: str = "float16"
    max_model_len: int = 512
    gpu_memory_utilization: float = 0.40
    enforce_eager: bool = True
    disable_custom_all_reduce: bool = True
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if (
            isinstance(self.tensor_parallel_size, bool)
            or not isinstance(self.tensor_parallel_size, int)
            or self.tensor_parallel_size < 1
        ):
            raise ValueError("tensor_parallel_size must be a positive integer")
        if self.model_representation not in {"transformers", "sharded_state"}:
            raise ValueError(
                "model_representation must be 'transformers' or 'sharded_state'"
            )
        if self.model_representation == "sharded_state" and self.load_format not in {
            None,
            "sharded_state",
        }:
            raise ValueError(
                "sharded_state representation requires load_format='sharded_state'"
            )
        if self.model_representation == "sharded_state" and self.load_format is None:
            object.__setattr__(self, "load_format", "sharded_state")
        if not self.dtype.strip():
            raise ValueError("dtype must be non-empty")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be positive")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        for name, value in (
            ("max_num_batched_tokens", self.max_num_batched_tokens),
            ("max_num_seqs", self.max_num_seqs),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer when set")
        if self.max_num_seqs is not None and self.max_num_seqs < len(
            self.workload.prompts
        ):
            raise ValueError(
                "max_num_seqs cannot be smaller than the workload request count"
            )

    @property
    def effective_load_format(self) -> str | None:
        if self.model_representation == "sharded_state":
            return "sharded_state"
        return self.load_format

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["workload"]["prompts"] = list(self.workload.prompts)
        data["load_format"] = self.effective_load_format
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BenchmarkSpec:
        return cls(
            model=str(data["model"]),
            tensor_parallel_size=int(data["tensor_parallel_size"]),
            model_revision=(
                str(data["model_revision"])
                if data.get("model_revision") is not None
                else None
            ),
            model_representation=str(
                data.get("model_representation", "transformers")
            ),
            load_format=(
                str(data["load_format"])
                if data.get("load_format") is not None
                else None
            ),
            dtype=str(data.get("dtype", "float16")),
            max_model_len=int(data.get("max_model_len", 512)),
            gpu_memory_utilization=float(data.get("gpu_memory_utilization", 0.40)),
            enforce_eager=bool(data.get("enforce_eager", True)),
            disable_custom_all_reduce=bool(
                data.get("disable_custom_all_reduce", True)
            ),
            max_num_batched_tokens=(
                int(data["max_num_batched_tokens"])
                if data.get("max_num_batched_tokens") is not None
                else None
            ),
            max_num_seqs=(
                int(data["max_num_seqs"])
                if data.get("max_num_seqs") is not None
                else None
            ),
            workload=WorkloadSpec.from_mapping(data.get("workload", {})),
        )


def build_benchmark_plan(spec: BenchmarkSpec, output: str | Path) -> dict[str, Any]:
    """Return a side-effect-free plan for an explicit benchmark invocation."""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "planned_not_executed",
        "measurement_mode": MEASUREMENT_MODE,
        "engine": spec.to_dict(),
        "evidence_output": str(Path(output).expanduser()),
        "operations": [
            "validate visible GPU count against tensor_parallel_size",
            "capture runtime fingerprint and nvidia-smi topology",
            "inspect sharded_state topology when selected",
            "construct one upstream vLLM offline LLM engine",
            f"run {spec.workload.warmup_runs} excluded warmup batch(es)",
            f"run {spec.workload.measurement_runs} measured batch(es)",
            "sample compact per-GPU nvidia-smi telemetry",
            "write one new JSON evidence file",
        ],
        "dry_run": True,
    }


def _safe_new_file(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    lexical = Path(os.path.abspath(expanded))
    resolved = expanded.resolve(strict=False)
    if lexical != resolved:
        raise BenchmarkError(
            f"refusing evidence path that traverses a symlink: {lexical} -> {resolved}"
        )
    kaggle_input = Path("/kaggle/input")
    if resolved == kaggle_input or kaggle_input in resolved.parents:
        raise BenchmarkError(f"refusing to write evidence under /kaggle/input: {resolved}")
    if resolved.exists():
        raise BenchmarkError(f"refusing to overwrite existing evidence file: {resolved}")
    if not resolved.parent.is_dir():
        raise BenchmarkError(
            f"evidence parent directory does not exist: {resolved.parent}"
        )
    return resolved


def prepare_evidence_directory(path: str | Path) -> Path:
    """Create one absent evidence directory after rejecting symlink traversal."""

    expanded = Path(path).expanduser()
    lexical = Path(os.path.abspath(expanded))
    resolved = expanded.resolve(strict=False)
    if lexical != resolved:
        raise BenchmarkError(
            f"refusing evidence directory that traverses a symlink: {lexical} -> {resolved}"
        )
    kaggle_input = Path("/kaggle/input")
    if resolved == kaggle_input or kaggle_input in resolved.parents:
        raise BenchmarkError(f"refusing to write evidence under /kaggle/input: {resolved}")
    if resolved.exists():
        raise BenchmarkError(
            f"refusing to reuse existing evidence directory: {resolved}"
        )
    if not resolved.parent.is_dir():
        raise BenchmarkError(
            f"evidence directory parent does not exist: {resolved.parent}"
        )
    resolved.mkdir()
    return resolved


def write_json_new(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write formatted JSON to an absent, non-symlinked file."""

    destination = _safe_new_file(path)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def summarize(values: Sequence[float]) -> dict[str, float | int | None]:
    """Return compact descriptive statistics for finite measurements."""

    normalized = [float(value) for value in values]
    if not normalized:
        raise ValueError("at least one measurement is required")
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("measurements must be finite")
    return {
        "count": len(normalized),
        "mean": statistics.fmean(normalized),
        "median": statistics.median(normalized),
        "sample_standard_deviation": (
            statistics.stdev(normalized) if len(normalized) > 1 else None
        ),
        "min": min(normalized),
        "max": max(normalized),
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_git_commit() -> str | None:
    package_file = Path(__file__).resolve()
    repository = next(
        (candidate for candidate in package_file.parents if (candidate / ".git").exists()),
        None,
    )
    if repository is None:
        return None
    try:
        result = subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def runtime_identity(environment: Environment) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build source/process identity and hardware sections from existing discovery."""

    libc_name, libc_version = platform.libc_ver()
    identity = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "kaggle_vllm_version": _distribution_version("kaggle-vllm") or "unknown",
        "vllm_distribution_version": _distribution_version("vllm"),
        "source_git_commit": _source_git_commit(),
        "python": environment.python,
        "os": platform.system(),
        "platform": environment.platform,
        "glibc": libc_version if libc_name == "glibc" else None,
        "measurement_mode": MEASUREMENT_MODE,
    }
    hardware = asdict(environment)
    hardware["gpu_count"] = environment.gpu_count
    hardware["cuda_toolkit"] = environment.nvcc_version
    return identity, hardware


def _load_vllm() -> tuple[type[Any], type[Any]]:
    try:
        module = importlib.import_module("vllm")
    except (ImportError, OSError):
        try:
            activate_runtime()
            module = importlib.import_module("vllm")
        except (ImportError, OSError, RuntimeError) as error:
            raise VLLMNotInstalledError(
                "vLLM is unavailable; bootstrap and activate the validated runtime "
                "before executing a GPU benchmark"
            ) from error
    try:
        return module.LLM, module.SamplingParams
    except AttributeError as error:
        raise VLLMNotInstalledError(
            "the activated vLLM module does not expose LLM and SamplingParams"
        ) from error


def _trial_metrics(outputs: Sequence[Any], elapsed: float, index: int) -> dict[str, Any]:
    input_tokens = sum(len(getattr(output, "prompt_token_ids", ()) or ()) for output in outputs)
    output_tokens = sum(
        len(getattr(candidate, "token_ids", ()) or ())
        for output in outputs
        for candidate in (getattr(output, "outputs", ()) or ())
    )
    requests = len(outputs)
    return {
        "index": index,
        "requests_completed": requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "wall_duration_seconds": elapsed,
        "aggregate_input_tokens_per_second": (
            input_tokens / elapsed if elapsed > 0 else None
        ),
        "aggregate_output_tokens_per_second": (
            output_tokens / elapsed if elapsed > 0 else None
        ),
        "request_completion_wall_seconds": [elapsed for _ in range(requests)],
    }


def aggregate_trials(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated offline batches without conflating engine load time."""

    if not trials:
        raise ValueError("at least one measured trial is required")
    durations = [float(trial["wall_duration_seconds"]) for trial in trials]
    input_rates = [float(trial["aggregate_input_tokens_per_second"]) for trial in trials]
    output_rates = [
        float(trial["aggregate_output_tokens_per_second"]) for trial in trials
    ]
    request_latencies = [
        float(value)
        for trial in trials
        for value in trial["request_completion_wall_seconds"]
    ]
    total_duration = sum(durations)
    total_input = sum(int(trial["input_tokens"]) for trial in trials)
    total_output = sum(int(trial["output_tokens"]) for trial in trials)
    return {
        "trial_wall_duration_seconds": summarize(durations),
        "trial_input_tokens_per_second": summarize(input_rates),
        "trial_output_tokens_per_second": summarize(output_rates),
        "request_completion_wall_seconds": summarize(request_latencies),
        "total_wall_duration_seconds": total_duration,
        "requests_completed": sum(int(trial["requests_completed"]) for trial in trials),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "aggregate_input_tokens_per_second": total_input / total_duration,
        "aggregate_output_tokens_per_second": total_output / total_duration,
    }


def run_offline_benchmark(
    spec: BenchmarkSpec,
    output: str | Path,
    *,
    environment: Environment | None = None,
    clock: Callable[[], float] = time.perf_counter,
    monitor_factory: Callable[[], GPUMonitor] = GPUMonitor,
    topology_capture: Callable[[], dict[str, Any]] = capture_topology,
) -> dict[str, Any]:
    """Execute one real offline vLLM benchmark and write schema-v1 evidence."""

    destination = _safe_new_file(output)
    runtime = environment or collect()
    validate_tensor_parallel_size(spec.tensor_parallel_size, runtime.gpu_count)
    llm_class, sampling_class = _load_vllm()
    identity, hardware = runtime_identity(runtime)
    topology = topology_capture()
    profile = load_profile()
    identity["native_runtime"] = {
        "source_tag": profile.vllm_source_tag,
        "source_commit": profile.vllm_source_commit,
        "wheel": profile.wheel_filename,
        "sha256": profile.wheel_sha256,
        "hf_repository": profile.hf_repo_id,
        "hf_revision": profile.hf_revision,
    }

    model_inspection: dict[str, Any] | None = None
    if spec.model_representation == "sharded_state":
        inspection = inspect_sharded_model(
            spec.model,
            expected_tensor_parallel_size=spec.tensor_parallel_size,
        )
        if not inspection.valid:
            raise BenchmarkError("sharded_state model inspection is not valid")
        model_inspection = inspection.to_dict()

    engine_options: dict[str, Any] = {
        "model": spec.model,
        "tensor_parallel_size": spec.tensor_parallel_size,
        "dtype": spec.dtype,
        "max_model_len": spec.max_model_len,
        "gpu_memory_utilization": spec.gpu_memory_utilization,
        "enforce_eager": spec.enforce_eager,
        "disable_custom_all_reduce": spec.disable_custom_all_reduce,
        "seed": spec.workload.seed,
    }
    model_is_local = Path(spec.model).expanduser().is_dir()
    optional_options = {
        "revision": spec.model_revision if not model_is_local else None,
        "load_format": spec.effective_load_format,
        "max_num_batched_tokens": spec.max_num_batched_tokens,
        "max_num_seqs": spec.max_num_seqs,
    }
    engine_options.update(
        {key: value for key, value in optional_options.items() if value is not None}
    )
    sampling = sampling_class(
        temperature=spec.workload.temperature,
        max_tokens=spec.workload.max_output_tokens,
        ignore_eos=spec.workload.ignore_eos,
        seed=spec.workload.seed,
    )

    trials: list[dict[str, Any]] = []
    load_started = clock()
    llm = llm_class(**engine_options)
    model_load_seconds = clock() - load_started
    with monitor_factory() as monitor:
        for _ in range(spec.workload.warmup_runs):
            llm.generate(list(spec.workload.prompts), sampling)
    for index in range(spec.workload.measurement_runs):
            started = clock()
            outputs = llm.generate(list(spec.workload.prompts), sampling)
            elapsed = clock() - started
            trials.append(_trial_metrics(outputs, elapsed, index))

    engine_evidence = spec.to_dict()
    engine_evidence.pop("workload")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "executed",
        "identity": identity,
        "hardware": hardware,
        "topology": topology,
        "engine": {
            **engine_evidence,
            "model_revision_applied_to_engine": bool(
                spec.model_revision and not model_is_local
            ),
            "expected_attention_backend_for_profile": profile.attention_backend,
            "observed_attention_backend": "not_measured",
            "model_inspection": model_inspection,
        },
        "workload": {
            **asdict(spec.workload),
            "prompts": list(spec.workload.prompts),
            "request_count_per_trial": len(spec.workload.prompts),
        },
        "measurements": {
            "model_load_wall_seconds": model_load_seconds,
            "warmup_runs_excluded": spec.workload.warmup_runs,
            "trials": trials,
            "aggregate": aggregate_trials(trials),
        },
        "gpu_telemetry": monitor.to_dict(),
        "limitations": [
            "Metrics describe offline LLM.generate batches, not HTTP serving.",
            (
                "Per-request completion wall time is the containing batch duration; "
                "individual completion timestamps are not exposed by this harness."
            ),
            "Sampled nvidia-smi telemetry can miss short-lived peaks.",
            "Throughput differences alone do not isolate NCCL or topology causality.",
            (
                "The expected attention backend is profile metadata; runtime logs "
                "remain the evidence for the backend actually selected."
            ),
        ],
    }
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
