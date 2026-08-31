"""Lightweight NVIDIA topology and telemetry capture for benchmark evidence."""

from __future__ import annotations

import re
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Self

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_NVLINK = re.compile(r"^NV\d+$")
_TELEMETRY_FIELDS = (
    "index",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
)


@dataclass(frozen=True)
class CommandCapture:
    """Serializable result of one non-mutating diagnostic command."""

    command: tuple[str, ...]
    status: str
    returncode: int | None
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPUSample:
    """One parsed ``nvidia-smi`` telemetry sample."""

    captured_at: str
    index: int
    memory_used_mib: float | None
    utilization_percent: float | None
    temperature_c: float | None
    power_draw_w: float | None
    sm_clock_mhz: float | None
    memory_clock_mhz: float | None


CommandRunner = Callable[[Sequence[str]], CommandCapture]


def run_command(
    command: Sequence[str],
    *,
    timeout: float = 15.0,
) -> CommandCapture:
    """Capture a command without a shell and preserve failures as observations."""

    normalized = tuple(str(part) for part in command)
    try:
        result = subprocess.run(
            normalized,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        return CommandCapture(normalized, "unavailable", None, "", str(error))
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        return CommandCapture(normalized, "timeout", None, stdout, stderr)
    except OSError as error:
        return CommandCapture(normalized, "error", None, "", str(error))
    return CommandCapture(
        command=normalized,
        status="ok" if result.returncode == 0 else "error",
        returncode=result.returncode,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


def _clean_topology(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def parse_topology_matrix(text: str) -> dict[str, Any]:
    """Parse GPU-to-GPU links from ``nvidia-smi topo -m`` conservatively."""

    lines = [line.strip() for line in _clean_topology(text).splitlines()]
    header_tokens: list[str] = []
    row_tokens: list[list[str]] = []
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        if not header_tokens and tokens[0].startswith("GPU"):
            gpu_prefix = []
            for token in tokens:
                if token.startswith("GPU") and token[3:].isdigit():
                    gpu_prefix.append(token)
                else:
                    break
            if len(gpu_prefix) >= 1:
                header_tokens = gpu_prefix
                continue
        if header_tokens and tokens[0] in header_tokens:
            row_tokens.append(tokens)

    links: list[dict[str, str]] = []
    rows = {tokens[0]: tokens[1:] for tokens in row_tokens}
    for left_index, left in enumerate(header_tokens):
        values = rows.get(left, [])
        for right_index in range(left_index + 1, len(header_tokens)):
            if right_index >= len(values):
                continue
            link = values[right_index]
            if link == "X":
                continue
            links.append(
                {"gpu_a": left, "gpu_b": header_tokens[right_index], "path": link}
            )

    nvlink_observed: bool | None
    if not links:
        nvlink_observed = None
    else:
        nvlink_observed = any(_NVLINK.fullmatch(link["path"]) for link in links)
    return {
        "gpu_labels": header_tokens,
        "links": links,
        "nvlink_observed": nvlink_observed,
    }


def capture_topology(*, runner: CommandRunner = run_command) -> dict[str, Any]:
    """Capture NVIDIA topology and peer-access observations without mutation."""

    matrix = runner(("nvidia-smi", "topo", "-m"))
    peer_read = runner(("nvidia-smi", "topo", "-p2p", "r"))
    peer_write = runner(("nvidia-smi", "topo", "-p2p", "w"))
    return {
        "matrix": matrix.to_dict(),
        "parsed_matrix": (
            parse_topology_matrix(matrix.stdout)
            if matrix.status == "ok"
            else {"gpu_labels": [], "links": [], "nvlink_observed": None}
        ),
        "peer_access_read": peer_read.to_dict(),
        "peer_access_write": peer_write.to_dict(),
        "interpretation": (
            "observed nvidia-smi output only; no DMA or causal performance claim"
        ),
    }


def _optional_float(value: str) -> float | None:
    normalized = value.strip()
    if normalized.casefold() in {"", "n/a", "na", "[not supported]", "not supported"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_telemetry_csv(text: str, *, captured_at: str) -> list[GPUSample]:
    """Parse the fixed telemetry query while tolerating unsupported fields."""

    samples: list[GPUSample] = []
    for line in text.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != len(_TELEMETRY_FIELDS):
            continue
        try:
            index = int(values[0])
        except ValueError:
            continue
        samples.append(
            GPUSample(
                captured_at=captured_at,
                index=index,
                memory_used_mib=_optional_float(values[1]),
                utilization_percent=_optional_float(values[2]),
                temperature_c=_optional_float(values[3]),
                power_draw_w=_optional_float(values[4]),
                sm_clock_mhz=_optional_float(values[5]),
                memory_clock_mhz=_optional_float(values[6]),
            )
        )
    return samples


def capture_gpu_sample(*, runner: CommandRunner = run_command) -> tuple[
    list[GPUSample], CommandCapture
]:
    """Capture one telemetry sample from every visible NVIDIA GPU."""

    command = (
        "nvidia-smi",
        f"--query-gpu={','.join(_TELEMETRY_FIELDS)}",
        "--format=csv,noheader,nounits",
    )
    captured_at = datetime.now(timezone.utc).isoformat()
    result = runner(command)
    samples = (
        parse_telemetry_csv(result.stdout, captured_at=captured_at)
        if result.status == "ok"
        else []
    )
    return samples, result


def _values(samples: list[GPUSample], attribute: str) -> list[float]:
    return [
        float(value)
        for sample in samples
        if (value := getattr(sample, attribute)) is not None
    ]


def summarize_gpu_samples(samples: list[GPUSample]) -> list[dict[str, Any]]:
    """Summarize samples without embedding a potentially large time series."""

    summaries: list[dict[str, Any]] = []
    for index in sorted({sample.index for sample in samples}):
        selected = [sample for sample in samples if sample.index == index]
        memory = _values(selected, "memory_used_mib")
        utilization = _values(selected, "utilization_percent")
        temperature = _values(selected, "temperature_c")
        power = _values(selected, "power_draw_w")
        sm_clock = _values(selected, "sm_clock_mhz")
        memory_clock = _values(selected, "memory_clock_mhz")
        summaries.append(
            {
                "index": index,
                "sample_count": len(selected),
                "first_sample_at": selected[0].captured_at,
                "last_sample_at": selected[-1].captured_at,
                "first_memory_used_mib": memory[0] if memory else None,
                "last_memory_used_mib": memory[-1] if memory else None,
                "peak_memory_used_mib": max(memory) if memory else None,
                "mean_utilization_percent": (
                    sum(utilization) / len(utilization) if utilization else None
                ),
                "peak_utilization_percent": max(utilization) if utilization else None,
                "peak_temperature_c": max(temperature) if temperature else None,
                "peak_power_draw_w": max(power) if power else None,
                "peak_sm_clock_mhz": max(sm_clock) if sm_clock else None,
                "peak_memory_clock_mhz": (
                    max(memory_clock) if memory_clock else None
                ),
            }
        )
    return summaries


class GPUMonitor:
    """Sample ``nvidia-smi`` in a daemon thread and retain compact summaries."""

    def __init__(
        self,
        interval_seconds: float = 0.25,
        *,
        runner: CommandRunner = run_command,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.interval_seconds = interval_seconds
        self.runner = runner
        self.samples: list[GPUSample] = []
        self.failures: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._collect, daemon=True)

    def _sample(self) -> None:
        samples, result = capture_gpu_sample(runner=self.runner)
        self.samples.extend(samples)
        if result.status != "ok":
            self.failures.append(result.to_dict())

    def _collect(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def __enter__(self) -> Self:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 3))
        self._sample()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "nvidia-smi",
            "sampling_interval_seconds": self.interval_seconds,
            "summaries": summarize_gpu_samples(self.samples),
            "capture_failures": self.failures,
            "raw_samples_retained": False,
        }
