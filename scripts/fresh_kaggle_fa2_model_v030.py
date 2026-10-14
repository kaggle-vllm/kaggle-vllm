"""Optional offline Qwen2.5 1.5B backend-selection workload on four L4s.

The runner evaluates backend logs afterward. Successful generation alone is
never reported as FA2-backed inference.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.model / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"offline model config unavailable: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("architectures") != ["Qwen2ForCausalLM"] or config.get("num_attention_heads", 0) % 4:
        raise RuntimeError("optional model is not the expected TP4-compatible Qwen2 architecture")
    from kaggle_vllm import KaggleLLM, KaggleSamplingParams, activate_runtime
    if not activate_runtime():
        raise RuntimeError("SDK runtime is not staged")
    import psutil
    process = psutil.Process()
    started = time.monotonic()
    llm = KaggleLLM(model=str(args.model), tensor_parallel_size=4, max_model_len=512, gpu_memory_utilization=0.50)
    initialized = time.monotonic()
    output = llm.generate(
        ["In one sentence, explain why multiple GPUs can share a matrix operation."],
        KaggleSamplingParams(temperature=0.0, max_tokens=48),
    )
    generated = time.monotonic()
    text = output[0].outputs[0].text.strip()
    tokens = list(output[0].outputs[0].token_ids)
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 8 or len(set(word.lower() for word in words)) < 6 or len(tokens) < 8:
        raise RuntimeError(f"optional model generation is not meaningful: {text!r}")
    backend = llm.effective_attention_backend()
    attention_config = getattr(getattr(llm.upstream.llm_engine, "vllm_config", None), "attention_config", None)
    configured_backend = getattr(attention_config, "backend", None)
    flash_attn_version = getattr(attention_config, "flash_attn_version", None)
    engine = llm.upstream.llm_engine
    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
    del output, llm, engine
    gc.collect()
    deadline = time.monotonic() + 30
    remaining: list[int] = []
    while time.monotonic() < deadline:
        remaining = [child.pid for child in process.children(recursive=True) if child.is_running() and child.status() != psutil.STATUS_ZOMBIE]
        if not remaining:
            break
        time.sleep(1)
    if remaining:
        raise RuntimeError(f"TP child workers remain: {remaining}")
    result = {
        "schema_version": 3,
        "generation_status": "PASS",
        "fa2_runtime_status": "NOT_PROVEN_UNTIL_LOG_REVIEW",
        "model": str(args.model),
        "architecture": config["architectures"][0],
        "attention_heads": config["num_attention_heads"],
        "tensor_parallel_size": 4,
        "text": text,
        "generated_tokens": len(tokens),
        "unique_generated_ids": len(set(tokens)),
        "engine_init_seconds": round(initialized - started, 3),
        "generation_seconds": round(generated - initialized, 3),
        "effective_backend_diagnostic": backend,
        "configured_backend_diagnostic": str(configured_backend) if configured_backend is not None else None,
        "flash_attn_version_diagnostic": flash_attn_version,
        "worker_cleanup": "PASS",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("OPTIONAL_MODEL_GENERATION_PASS; backend proof requires the saved vLLM runtime log", flush=True)


if __name__ == "__main__":
    main()
