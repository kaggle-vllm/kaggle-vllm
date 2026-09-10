"""Parse preserved periodic vLLM scheduler signals without inferring batch size."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .errors import ResearchEvidenceError

IDENTITY = re.compile(r"qwen-tp(?P<tp>[12])-c(?P<concurrency>\d+)\.server\.log$")
SIGNAL = re.compile(
    r"Running:\s*(?P<running>\d+)\s+reqs,\s*"
    r"Waiting:\s*(?P<waiting>\d+)\s+reqs,\s*"
    r"GPU KV cache usage:\s*(?P<kv>[0-9]+(?:\.[0-9]+)?)%"
)


def parse_m2_scheduler_signals(m2_dir: str | Path) -> dict[str, Any]:
    """Parse all twelve required M2 server logs and fail on malformed candidates."""

    root = Path(m2_dir)
    expected = {
        (tp, concurrency)
        for tp in (1, 2)
        for concurrency in (1, 4, 8, 16, 32, 64)
    }
    seen: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("qwen-tp*-c*.server.log")):
        identity = IDENTITY.match(path.name)
        if identity is None:
            raise ResearchEvidenceError(f"unrecognized M2 server-log filename: {path.name}")
        key = (int(identity["tp"]), int(identity["concurrency"]))
        if key not in expected or key in seen:
            raise ResearchEvidenceError(f"unexpected or duplicate M2 server log: {path.name}")
        seen.add(key)
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if "[loggers.py:" not in line or "Running:" not in line:
                continue
            signal = SIGNAL.search(line)
            if signal is None:
                raise ResearchEvidenceError(
                    f"malformed required scheduler signal {path.name}:{line_number}"
                )
            rows.append(
                {
                    "source_file": path.name,
                    "source_line": line_number,
                    "tensor_parallel_size": key[0],
                    "request_concurrency_setting": key[1],
                    "running_requests": int(signal["running"]),
                    "waiting_requests": int(signal["waiting"]),
                    "gpu_kv_cache_usage_percent": float(signal["kv"]),
                    "evidence": "OBSERVED_PERIODIC_SERVER_LOG",
                }
            )
    if seen != expected:
        raise ResearchEvidenceError(f"M2 scheduler audit missing logs: {sorted(expected - seen)}")
    if not rows:
        raise ResearchEvidenceError("M2 scheduler audit found no periodic signals")
    return {
        "schema_version": "kaggle-vllm-m2-scheduler-signals-v1",
        "status": "PARSED_FROM_PRESERVED_LOGS",
        "rows": rows,
        "instantaneous_decode_batch": "unobserved",
        "interpretation": (
            "Running requests are periodic server-log observations; they are not "
            "per-step decode batch size or token batch composition."
        ),
    }
