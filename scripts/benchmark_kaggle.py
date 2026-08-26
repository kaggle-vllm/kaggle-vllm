"""Kaggle GPU benchmark harness for controlled TP=1/TP=2 comparisons."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self


@dataclass
class GPUSample:
    timestamp: float
    index: int
    memory_used_mib: int
    utilization_percent: int


class GPUMonitor:
    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.samples: list[GPUSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._collect, daemon=True)

    def _collect(self) -> None:
        while not self._stop.is_set():
            command = [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    try:
                        index, memory, utilization = (
                            int(value.strip()) for value in line.split(",")
                        )
                    except (TypeError, ValueError):
                        continue
                    self.samples.append(
                        GPUSample(time.time(), index, memory, utilization)
                    )
            self._stop.wait(self.interval)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval * 3))

    def summary(self) -> list[dict[str, int]]:
        result = []
        for index in sorted({sample.index for sample in self.samples}):
            samples = [sample for sample in self.samples if sample.index == index]
            result.append(
                {
                    "index": index,
                    "peak_memory_used_mib": max(
                        sample.memory_used_mib for sample in samples
                    ),
                    "peak_utilization_percent": max(
                        sample.utilization_percent for sample in samples
                    ),
                    "last_memory_used_mib": samples[-1].memory_used_mib,
                }
            )
        return result


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def result_schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "pending_gpu_execution",
        "configuration": {},
        "environment": {},
        "topology": {},
        "runs": [],
        "aggregate": {},
    }


def _metrics(output: Any, elapsed: float) -> dict[str, float | int | None]:
    prompt_tokens = len(output.prompt_token_ids or ())
    output_tokens = sum(len(item.token_ids) for item in output.outputs)
    request_metrics = getattr(output, "metrics", None)
    first = getattr(request_metrics, "first_token_ts", 0.0) or 0.0
    arrival = getattr(request_metrics, "arrival_time", 0.0) or 0.0
    last = getattr(request_metrics, "last_token_ts", 0.0) or 0.0
    ttft = first - arrival if first > arrival > 0 else None
    engine_e2e = last - arrival if last > arrival > 0 else None
    decode = last - first if last > first > 0 else None
    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "wall_latency_seconds": elapsed,
        "engine_latency_seconds": engine_e2e,
        "time_to_first_token_seconds": ttft,
        "prefill_tokens_per_second": (
            prompt_tokens / ttft if ttft and ttft > 0 else None
        ),
        "generation_tokens_per_second": (
            output_tokens / decode if decode and decode > 0 else None
        ),
        "end_to_end_tokens_per_second": (
            output_tokens / elapsed if elapsed > 0 else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument(
        "--tensor-parallel-size", type=int, choices=(1, 2), required=True
    )
    parser.add_argument("--prompt", default="Explain tensor parallel inference.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    parser.add_argument(
        "--enforce-eager", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-schema", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.print_schema:
        print(json.dumps(result_schema(), indent=2))
        return 0
    if args.repeats < 1 or args.max_tokens < 1:
        raise SystemExit("repeats and max-tokens must be positive")

    from vllm import LLM, SamplingParams

    from kaggle_vllm.environment import collect

    environment = collect()
    if environment.gpu_count < args.tensor_parallel_size:
        raise SystemExit("visible GPU count is smaller than tensor parallel size")

    config = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "prompt": args.prompt,
        "requested_output_tokens": args.max_tokens,
        "repeats": args.repeats,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
    }
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    llm.generate([args.prompt], sampling)

    runs = []
    with GPUMonitor() as monitor:
        for index in range(args.repeats):
            started = time.perf_counter()
            outputs = llm.generate([args.prompt], sampling)
            elapsed = time.perf_counter() - started
            runs.append({"index": index, **_metrics(outputs[0], elapsed)})

    latencies = [float(run["wall_latency_seconds"]) for run in runs]
    throughputs = [float(run["end_to_end_tokens_per_second"]) for run in runs]
    payload = result_schema()
    payload.update(
        {
            "status": "executed",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "configuration": config,
            "environment": asdict(environment),
            "topology": {
                "nvidia_smi": _command_output(["nvidia-smi"]),
                "nvidia_smi_topology": _command_output(["nvidia-smi", "topo", "-m"]),
                "gpu_monitor": monitor.summary(),
            },
            "runs": runs,
            "aggregate": {
                "mean_wall_latency_seconds": sum(latencies) / len(latencies),
                "mean_output_tokens_per_second": sum(throughputs) / len(throughputs),
            },
        }
    )
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
