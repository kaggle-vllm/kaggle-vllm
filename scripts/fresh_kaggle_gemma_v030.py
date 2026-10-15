"""Fresh Kaggle L4x4 Gemma semantic acceptance using the public SDK wrapper.

Execute only after offline bootstrap and environment activation on CPython 3.12.
This script is an acceptance gate, not evidence that a fresh run has passed.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import time
from pathlib import Path

MODEL = "/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2"
SHUTDOWN_TIMEOUT = 30.0
WORKER_EXIT_TIMEOUT = 30.0


def meaningful_text(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text.lower())
    return len(words) >= 10 and len(set(words)) >= 7 and any(x in words for x in ("tensor", "tensors")) and any(x in words for x in ("gpu", "gpus", "parallel", "distributed"))


def child_snapshot(parent, psutil_module) -> list[dict]:
    """Capture full process identity because a numeric PID may be reused."""
    result = []
    for child in parent.children(recursive=True):
        try:
            with child.oneshot():
                status = child.status()
                start_time = child.create_time()
                try:
                    ppid = child.ppid()
                except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
                    ppid = None
                try:
                    name = child.name()
                except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
                    name = "[unavailable]"
                try:
                    cmdline = [] if status == psutil_module.STATUS_ZOMBIE else child.cmdline()
                except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
                    cmdline = []
                result.append({
                    "pid": child.pid,
                    "ppid": ppid,
                    "status": status,
                    "start_time": start_time,
                    "name": name,
                    "cmdline": cmdline,
                })
        except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
            continue
    return sorted(result, key=lambda item: (item["pid"], item["start_time"]))


def gpu_process_snapshot() -> dict:
    """Observe CUDA contexts independently; retain an error if unavailable."""
    command = ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": repr(exc), "processes": []}
    if result.returncode:
        return {"available": False, "error": result.stderr.strip(), "processes": []}
    processes = []
    for line in result.stdout.splitlines():
        columns = [part.strip() for part in line.split(",")]
        if columns and columns[0].isdigit():
            processes.append({"pid": int(columns[0]), "used_gpu_memory_mib": columns[1] if len(columns) > 1 else None})
    return {"available": True, "processes": processes}


def classify_children(before: list[dict], after: list[dict], baseline: list[dict], gpu: dict, zombie_status: str) -> tuple[list[dict], list[int]]:
    baseline_ids = {(item["pid"], item["start_time"]) for item in baseline}
    old_starts = {item["pid"]: item["start_time"] for item in before}
    gpu_pids = {item["pid"] for item in gpu.get("processes", [])}
    classified = []
    unresolved = []
    for item in after:
        identity = (item["pid"], item["start_time"])
        marker = (item["name"] + " " + " ".join(item["cmdline"])).lower()
        if item["status"] == zombie_status:
            role = "ZOMBIE_NOT_REAPED"
        elif item["pid"] in old_starts and old_starts[item["pid"]] != item["start_time"]:
            # The prior identity exited. Still fail a *new* worker or GPU
            # owner rather than treating every reused PID as unrelated.
            role = "LIVE_WORKER" if ("worker_tp" in marker or "vllm::worker" in marker or item["pid"] in gpu_pids) else "UNRELATED_PROCESS"
        elif identity in baseline_ids:
            role = "UNRELATED_PROCESS"
        elif "worker_tp" in marker or "vllm::worker" in marker:
            role = "LIVE_WORKER"
        elif "enginecore" in marker or "engine_core" in marker:
            role = "SUPERVISOR_PROCESS"
        elif "resource_tracker" in marker or "forkserver" in marker:
            role = "SUPERVISOR_PROCESS"
        else:
            role = "UNKNOWN"
        if item["pid"] in gpu_pids and role not in ("ZOMBIE_NOT_REAPED", "UNRELATED_PROCESS"):
            role = "LIVE_WORKER"
        annotated = {**item, "classification": role, "gpu_context_observed": item["pid"] in gpu_pids}
        classified.append(annotated)
        # An unidentified owned child is unresolved, never silently accepted.
        # A resource tracker without a CUDA context is not a TP worker.
        if role in ("LIVE_WORKER", "UNKNOWN") or (role == "SUPERVISOR_PROCESS" and "resource_tracker" not in marker and "forkserver" not in marker):
            unresolved.append(item["pid"])
    return classified, sorted(set(unresolved))


def shutdown_and_verify(llm, parent, baseline: list[dict], psutil_module, *, timeout: float = WORKER_EXIT_TIMEOUT, sleep=time.sleep, clock=time.monotonic) -> dict:
    """Call vLLM 0.30.0 EngineCoreClient.shutdown and verify owned workers."""
    before = child_snapshot(parent, psutil_module)
    diagnostics = {"before_shutdown": before, "shutdown_method": "LLMEngine.engine_core.shutdown(timeout=30)", "snapshots": []}
    engine = llm.upstream.llm_engine
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if not callable(shutdown):
        raise RuntimeError("vLLM EngineCoreClient.shutdown is unavailable; cannot verify cleanup")
    # LLMEngine itself has no shutdown() in the frozen vLLM 0.30.0 wheel.
    # MPClient.shutdown stops its engine manager and releases IPC resources.
    shutdown(timeout=SHUTDOWN_TIMEOUT)
    diagnostics["explicit_shutdown_completed"] = True
    deadline = clock() + timeout
    while True:
        after = child_snapshot(parent, psutil_module)
        gpu = gpu_process_snapshot()
        classified, unresolved = classify_children(before, after, baseline, gpu, psutil_module.STATUS_ZOMBIE)
        after_ids = {(item["pid"], item["start_time"]) for item in after}
        terminated = [{**item, "classification": "TERMINATED"} for item in before if (item["pid"], item["start_time"]) not in after_ids]
        # A TP process might be reparented before this helper inspects its
        # descendants. Keep its original identity and check its CUDA context.
        before_by_pid = {item["pid"]: item["start_time"] for item in before}
        orphan_gpu_pids = []
        for item in gpu.get("processes", []):
            pid = item["pid"]
            if pid not in before_by_pid or any(row["pid"] == pid for row in after):
                continue
            try:
                if psutil_module.Process(pid).create_time() == before_by_pid[pid]:
                    orphan_gpu_pids.append(pid)
            except (psutil_module.NoSuchProcess, psutil_module.AccessDenied):
                continue
        unresolved = sorted(set(unresolved + orphan_gpu_pids))
        diagnostics["snapshots"].append({"children": classified, "terminated": terminated, "gpu": gpu, "orphan_gpu_pids": orphan_gpu_pids, "unresolved_pids": unresolved})
        if not unresolved:
            diagnostics["status"] = "PASS"
            return diagnostics
        if clock() >= deadline:
            diagnostics["status"] = "FAIL"
            diagnostics["unresolved_pids"] = unresolved
            return diagnostics
        sleep(min(1.0, max(0.0, deadline - clock())))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(MODEL))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/kaggle-vllm-v030-gemma-acceptance.json"))
    args = parser.parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model directory unavailable: {args.model}")
    from kaggle_vllm import KaggleLLM, KaggleSamplingParams, activate_runtime
    if not activate_runtime():
        raise SystemExit("SDK runtime is not staged; run kaggle-vllm bootstrap first")
    import psutil
    process = psutil.Process()
    baseline = child_snapshot(process, psutil)
    report = {
        "schema_version": 4,
        "status": "FAIL",
        "model": str(args.model),
        "tensor_parallel_size": 4,
        "renderer": "KaggleLLM.chat AutoProcessor/AutoTokenizer template, then generate",
        "worker_cleanup": "NOT_RUN",
    }
    llm = None
    output = None
    original_error = None
    original_traceback = None
    started = time.monotonic()
    try:
        llm = KaggleLLM(model=str(args.model), profile="kaggle-l4x4-cu129", tensor_parallel_size=4, dtype="bfloat16", gpu_memory_utilization=0.86, max_model_len=2048)
        initialized = time.monotonic()
        output = llm.chat(
            [{"role": "user", "content": "Explain tensor parallelism in one sentence."}],
            local_files_only=True,
            sampling_params=KaggleSamplingParams(temperature=0.0, max_tokens=64),
        )
        generated = time.monotonic()
        text = output[0].outputs[0].text.strip()
        token_ids = list(output[0].outputs[0].token_ids)
        report.update({
            "text": text,
            "meaningful": meaningful_text(text),
            "generated_tokens": len(token_ids),
            "unique_generated_ids": len(set(token_ids)),
            "engine_init_seconds": round(initialized - started, 3),
            "generation_seconds": round(generated - initialized, 3),
            "effective_backend_diagnostic": llm.effective_attention_backend(),
        })
        if not report["meaningful"]:
            raise RuntimeError(f"Gemma semantic response did not meet the acceptance gate: {text!r}")
    except BaseException as exc:
        original_error, original_traceback = exc, exc.__traceback__
        report["inference_error"] = repr(exc)
    finally:
        if llm is not None:
            try:
                report["cleanup_before_shutdown_snapshot"] = {
                    "children": child_snapshot(process, psutil), "gpu": gpu_process_snapshot()
                }
                cleanup = shutdown_and_verify(llm, process, baseline, psutil)
                report["cleanup_diagnostics"] = cleanup
                report["worker_cleanup"] = cleanup["status"]
                if cleanup["status"] != "PASS" and original_error is None:
                    original_error = RuntimeError(f"vLLM engine cleanup left unresolved processes: {cleanup['unresolved_pids']}")
            except BaseException as exc:
                report["worker_cleanup"] = "FAIL"
                report["cleanup_error"] = repr(exc)
                try:
                    report["cleanup_exception_snapshot"] = {
                        "children": child_snapshot(process, psutil), "gpu": gpu_process_snapshot()
                    }
                except BaseException as snapshot_error:
                    report["cleanup_exception_snapshot"] = {"capture_error": repr(snapshot_error)}
                if original_error is None:
                    original_error, original_traceback = exc, exc.__traceback__
        output = None
        llm = None
        gc.collect()
        report["status"] = "PASS" if original_error is None and report["worker_cleanup"] == "PASS" else "FAIL"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    if original_error is not None:
        raise original_error.with_traceback(original_traceback)


if __name__ == "__main__":
    main()
