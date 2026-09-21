"""Kaggle dual-T4 resource guards without CUDA import side effects."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ResearchEvidenceError

GPU_MEMORY_LIMIT_MIB = 14.5 * 1024
SYSTEM_RAM_LIMIT_BYTES = 28 * 1024**3
# Kaggle's practical ceiling is documented as decimal GB, not GiB.
WORKING_STORAGE_LIMIT_BYTES = 20_000_000_000


def directory_size(path: str | Path) -> int:
    """Measure regular-file bytes without following symlinks."""

    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [
            name for name in directories if not (Path(current) / name).is_symlink()
        ]
        for name in files:
            candidate = Path(current) / name
            if not candidate.is_symlink():
                total += candidate.stat().st_size
    return total


def require_disk_budget(
    working_root: str | Path,
    *,
    projected_additional_bytes: int,
    limit_bytes: int = WORKING_STORAGE_LIMIT_BYTES,
) -> dict[str, int]:
    """Refuse an operation that would exceed used-space or free-space budgets."""

    if projected_additional_bytes < 0:
        raise ResearchEvidenceError("projected disk addition cannot be negative")
    root = Path(working_root).resolve()
    if not root.is_dir():
        raise ResearchEvidenceError(f"working root does not exist: {root}")
    currently_used = directory_size(root)
    free = shutil.disk_usage(root).free
    projected = currently_used + projected_additional_bytes
    if projected > limit_bytes:
        raise ResearchEvidenceError(
            f"projected working storage {projected} exceeds limit {limit_bytes}"
        )
    if projected_additional_bytes > free:
        raise ResearchEvidenceError(
            f"projected addition {projected_additional_bytes} exceeds free bytes {free}"
        )
    return {
        "working_bytes": currently_used,
        "projected_additional_bytes": projected_additional_bytes,
        "projected_working_bytes": projected,
        "free_bytes": free,
        "limit_bytes": limit_bytes,
    }


def _system_memory() -> tuple[int, int]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        process = psutil.Process()
        descendants = process.children(recursive=True)
        process_rss = process.memory_info().rss
        for child in descendants:
            try:
                process_rss += child.memory_info().rss
            except psutil.Error:
                continue
        return int(memory.used), int(process_rss)
    except (ImportError, OSError):
        values: dict[str, int] = {}
        with Path("/proc/meminfo").open(encoding="utf-8") as handle:
            for line in handle:
                key, raw = line.split(":", maxsplit=1)
                values[key] = int(raw.strip().split()[0]) * 1024
        return values["MemTotal"] - values["MemAvailable"], -1


def _optional_float(value: str) -> float | None:
    if value.strip().casefold() in {"", "n/a", "[not supported]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class ResourceSample:
    """One system and GPU resource observation."""

    captured_at_utc: str
    phase: str
    measurement_active: bool
    system_used_bytes: int
    process_tree_rss_bytes: int
    gpu_index: int | None
    gpu_uuid: str | None
    gpu_name: str | None
    memory_used_mib: float | None
    memory_total_mib: float | None
    utilization_gpu_percent: float | None
    temperature_c: float | None
    power_draw_w: float | None
    sm_clock_mhz: float | None
    memory_clock_mhz: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_resource_samples(
    *, phase: str, measurement_active: bool
) -> list[ResourceSample]:
    """Capture all visible GPUs plus host/process memory."""

    system_used, process_rss = _system_memory()
    captured_at = datetime.now(timezone.utc).isoformat()
    command = (
        "nvidia-smi",
        (
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,"
            "temperature.gpu,power.draw,clocks.sm,clocks.mem"
        ),
        "--format=csv,noheader,nounits",
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return [
            ResourceSample(
                captured_at,
                phase,
                measurement_active,
                system_used,
                process_rss,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        ]
    samples: list[ResourceSample] = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 10:
            continue
        samples.append(
            ResourceSample(
                captured_at,
                phase,
                measurement_active,
                system_used,
                process_rss,
                int(row[0]),
                row[1].strip(),
                row[2].strip(),
                _optional_float(row[3]),
                _optional_float(row[4]),
                _optional_float(row[5]),
                _optional_float(row[6]),
                _optional_float(row[7]),
                _optional_float(row[8]),
                _optional_float(row[9]),
            )
        )
    return samples


class ResourceMonitor:
    """Sample resources and fail when a hard Kaggle guard is crossed."""

    def __init__(
        self,
        *,
        phase_provider: Callable[[], tuple[str, bool]],
        interval_seconds: float = 0.25,
        gpu_limit_mib: float = GPU_MEMORY_LIMIT_MIB,
        ram_limit_bytes: int = SYSTEM_RAM_LIMIT_BYTES,
        violation_callback: Callable[[str], None] | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.phase_provider = phase_provider
        self.interval_seconds = interval_seconds
        self.gpu_limit_mib = gpu_limit_mib
        self.ram_limit_bytes = ram_limit_bytes
        self.violation_callback = violation_callback
        self.samples: list[ResourceSample] = []
        self.violation: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._violation_reported = False

    def _sample_once(self) -> None:
        phase, active = self.phase_provider()
        samples = capture_resource_samples(phase=phase, measurement_active=active)
        self.samples.extend(samples)
        for sample in samples:
            if sample.system_used_bytes > self.ram_limit_bytes:
                self.violation = (
                    f"system RAM {sample.system_used_bytes} exceeded "
                    f"{self.ram_limit_bytes} bytes"
                )
            if (
                sample.memory_used_mib is not None
                and sample.memory_used_mib > self.gpu_limit_mib
            ):
                self.violation = (
                    f"GPU {sample.gpu_index} memory {sample.memory_used_mib} MiB "
                    f"exceeded {self.gpu_limit_mib} MiB"
                )
        if (
            self.violation is not None
            and not self._violation_reported
            and self.violation_callback is not None
        ):
            self._violation_reported = True
            self.violation_callback(self.violation)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("resource monitor already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4))
        self._sample_once()

    def raise_if_violated(self) -> None:
        if self.violation is not None:
            raise ResearchEvidenceError(self.violation)
