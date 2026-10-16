"""Run a small, evidence-producing trial with the organizer's ADK harness.

This uses the competition's vLLM server. It does not route the scored agent
through kaggle-vllm or replace the organizer's serving interface.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


DATA = Path("/kaggle/input/competitions/gemma-4-developer-agent")
MODEL_NAME = "gemma-4-31b-it-qat-w4a16-ct"
MODEL = Path("/kaggle/input/models/google/gemma-4/other") / MODEL_NAME / "2"
SUBMISSION_SHA256 = "f3836a74dc06c850b589112e14a06496928a44dabdfea92346ee794cffa0ab11"
SUBMISSION_FILES = {
    "agent.yaml": "5620046a60cc5f68e7c2871a3132979b17abc151e596c702cace062664393072",
    "configs/sampling.yaml": "8ffd66a1224a0d373765a9c1e283f2e420be1c7db0c296b482d02e2448c469dd",
    "eval_config.yaml": "427a48dc10d1e52b5b9a8c41eebc8f70bb1b8efa8dbbb14d4836538aa1fd6458",
    "prompts/system.md": "f08fcad6202a1e2867d18ad53ed86ecdd2130758a3fff64ab8d55ba40c6c22fa",
}
OUT = Path("/kaggle/working/06_agent_task_evaluation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_submission(source: Path, destination: Path) -> None:
    """Verify the exact ZIP members Kaggle unpacked on Dataset upload."""
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError("private agent Input is unavailable")
    if destination.exists():
        raise RuntimeError("use a fresh Kaggle session for this agent trial")
    actual = {str(path.relative_to(source)): path for path in source.rglob("*") if path.is_file()}
    if set(actual) != set(SUBMISSION_FILES):
        raise RuntimeError(f"agent Input file set differs: {sorted(actual)}")
    for name, expected in SUBMISSION_FILES.items():
        if actual[name].is_symlink() or sha256(actual[name]) != expected:
            raise RuntimeError(f"agent Input file identity differs: {name}")
    shutil.copytree(source, destination, symlinks=False)


def choose_tasks(tasks: list, count: int = 2) -> list:
    """Freeze the first public task from each distinct repository."""
    chosen = []
    seen = set()
    for task in tasks:
        if task.repo not in seen:
            chosen.append(task)
            seen.add(task.repo)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(f"need {count} distinct public task repositories")
    return chosen


def main() -> None:
    if sys.version_info[:2] != (3, 12) or not Path("/kaggle/input").is_dir():
        raise RuntimeError("trial requires a Kaggle Python 3.12 competition session")
    if not DATA.joinpath("tasks.jsonl").is_file() or not MODEL.joinpath("config.json").is_file():
        raise RuntimeError("official competition data or attached Gemma model is missing")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("trial must run with Internet disabled and offline model loading")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=20,
    ).stdout.splitlines()
    if [row.strip() for row in gpu] != ["NVIDIA L4, 8.9"] * 4:
        raise RuntimeError(f"requires four L4/SM89 GPUs: {gpu}")

    import litellm
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters, validate_directory
    from adk_submission.discovery import discover_declared_models
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks

    litellm.drop_params = True  # Required by the organizer's ADK bridge.
    source = Path("/kaggle/input/datasets/waqasm86/gemma4-agent-v031-private-input/submission")
    agent_dir = OUT / "submission"
    stage_submission(source, agent_dir)
    limits, generation_constraints = build_submission_limits()
    validate_directory(agent_dir, limits=limits)
    declared = discover_declared_models(agent_dir)
    if declared != {MODEL_NAME}:
        raise RuntimeError(f"submission must declare only {MODEL_NAME}: {declared}")
    tasks = choose_tasks(load_tasks(DATA / "tasks.jsonl"))
    OUT.mkdir(parents=True, exist_ok=True)
    selection = {
        "selection_rule": "first public task from each distinct repository in tasks.jsonl order",
        "tasks": [{"id": task.instance_id, "repo": task.repo, "base_commit": task.base_commit} for task in tasks],
        "submission_sha256": SUBMISSION_SHA256,
        "submission_file_sha256": SUBMISSION_FILES,
        "model_config_sha256": sha256(MODEL / "config.json"),
        "gpus": gpu,
        "torch": torch.__version__,
    }
    (OUT / "task_selection.json").write_text(json.dumps(selection, indent=2) + "\n")

    adapters = discover_adapters(agent_dir)
    if getattr(adapters, "adapters", None):
        raise RuntimeError("this exact trial submission does not include adapters")
    server = VllmServer(VllmConfig(
        model=str(MODEL), host="127.0.0.1", port=8000,
        tool_call_parser="gemma4", reasoning_parser="gemma4",
        max_model_len=32768, dtype="bfloat16", tensor_parallel_size=4,
        gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
        scheduler_reserve_full_isl=False, startup_timeout=1200,
    ), adapter_manifest=adapters)
    records = []
    started = time.monotonic()
    try:
        server.start()
        models = server.create_model_registry(aliases=[MODEL_NAME], model_prefix="openai/", api_key="EMPTY")
        config = EvalConfig(
            tasks_path=DATA / "tasks.jsonl", snapshots_dir=DATA / "snapshots",
            results_dir=OUT / "results", submission_dir=agent_dir,
            models=models, sandbox="subprocess", timeout_seconds=120,
            verification_timeout_seconds=300, max_tool_calls=40,
            max_time_minutes=5, limits=limits,
            generation_constraints=generation_constraints,
            adapter_manifest=adapters,
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(
                compaction_interval=15, overlap_size=2, token_threshold=14336,
                event_retention_size=5,
            ),
            graph_dir=str(DATA / "graphs"), embeddings_dir=str(DATA / "embeddings"),
            wheels_dir=DATA / "wheels", verbose=False,
        )
        evaluator = Evaluator(config)
        for index, task in enumerate(tasks, start=1):
            result = asyncio.run(evaluator.evaluate_task(task=task, task_index=index, total_tasks=len(tasks)))
            patch = result.agent_patch or ""
            (OUT / f"task_{index}.patch").write_text(patch)
            (OUT / f"task_{index}_test.log").write_text(result.test_output or "")
            record = {
                "id": task.instance_id, "repo": task.repo, "resolved": result.resolved,
                "test_exit_code": result.test_exit_code,
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "patch_bytes": len(patch.encode()), "tool_calls": result.tool_calls,
                "duration_seconds": result.duration_seconds, "error": result.error,
            }
            records.append(record)
            (OUT / "task_results.json").write_text(json.dumps(records, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    finally:
        server.stop()
        (OUT / "run_status.json").write_text(json.dumps({
            "tasks_finished": len(records), "tasks_requested": len(tasks),
            "elapsed_seconds": time.monotonic() - started,
            "server_stopped": True,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
