#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,ENV,GRAPH")

from kaggle_vllm import KaggleLLM, activate_runtime

manifest = os.environ.get("KAGGLE_VLLM_MANIFEST")
if not manifest or not activate_runtime(manifest):
    raise RuntimeError("kaggle-vllm runtime activation failed")

import torch
import vllm
from vllm import SamplingParams
from vllm.inputs import TokensPrompt


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model-commit", required=True)
    p.add_argument("--expected-internal-version", required=True)
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--core-prompt-tokens", type=int, default=128)
    p.add_argument("--core-output-tokens", type=int, default=32)
    p.add_argument("--capacity-output-tokens", type=int, default=32)
    p.add_argument("--capacity-margin-tokens", type=int, default=64)
    p.add_argument("--concurrency-levels", default="1,2,4,8")
    p.add_argument("--concurrency-prompt-tokens", type=int, default=128)
    p.add_argument("--concurrency-output-tokens", type=int, default=16)
    p.add_argument("--result-json", required=True)
    return p.parse_args()


def percentile(values, q):
    values = [float(x) for x in values if x is not None]
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def exact_ids(tokenizer, seed_text, target):
    base = tokenizer.encode(seed_text, add_special_tokens=False)
    if not base:
        raise RuntimeError("seed text tokenized to zero tokens")
    bos = tokenizer.bos_token_id
    ids = ([bos] if bos is not None else [])
    need = target - len(ids)
    if need < 0:
        return ids[:target]
    repeats = (need + len(base) - 1) // len(base)
    ids.extend((base * repeats)[:need])
    if len(ids) != target:
        raise RuntimeError((len(ids), target))
    return ids


def request_metrics(out, output_tokens):
    m = getattr(out, "metrics", None)
    if m is None:
        return {"ttft_s": None, "e2e_s": None, "tpot_s": None, "queue_s": None}
    arrival = getattr(m, "arrival_time", None)
    first = getattr(m, "first_token_time", None)
    finished = getattr(m, "finished_time", None)
    queue = getattr(m, "time_in_queue", None)
    ttft = (first - arrival) if arrival is not None and first is not None else None
    e2e = (finished - arrival) if arrival is not None and finished is not None else None
    tpot = None
    if first is not None and finished is not None and output_tokens > 1:
        tpot = (finished - first) / (output_tokens - 1)
    return {"ttft_s": ttft, "e2e_s": e2e, "tpot_s": tpot, "queue_s": queue}


def run_batch(llm, tokenizer, label, language, prompt_tokens, output_tokens, concurrency, repeat):
    seed = (
        "Efficient inference requires careful measurement of latency throughput memory and scheduling. "
        if language == "english"
        else
        "يتطلب الاستدلال الفعال قياساً دقيقاً لزمن الاستجابة والإنتاجية والذاكرة والجدولة. "
    )
    ids = exact_ids(tokenizer, seed, prompt_tokens)
    prompts = [TokensPrompt(prompt_token_ids=list(ids)) for _ in range(concurrency)]
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
        seed=0,
    )
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0

    total_prompt = sum(len(o.prompt_token_ids or []) for o in outs)
    total_output = sum(len(o.outputs[0].token_ids or []) for o in outs)
    per_req = [request_metrics(o, len(o.outputs[0].token_ids or [])) for o in outs]

    ttft = [x["ttft_s"] for x in per_req if x["ttft_s"] is not None]
    e2e = [x["e2e_s"] for x in per_req if x["e2e_s"] is not None]
    tpot = [x["tpot_s"] for x in per_req if x["tpot_s"] is not None]
    queue = [x["queue_s"] for x in per_req if x["queue_s"] is not None]

    return {
        "label": label,
        "language": language,
        "repeat": repeat,
        "concurrency": concurrency,
        "prompt_tokens_per_request": prompt_tokens,
        "requested_output_tokens_per_request": output_tokens,
        "request_count": len(outs),
        "total_prompt_tokens": total_prompt,
        "total_output_tokens": total_output,
        "wall_s": wall,
        "output_tokens_per_s": total_output / wall if wall else None,
        "total_tokens_per_s": (total_prompt + total_output) / wall if wall else None,
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p95_s": percentile(ttft, 0.95),
        "e2e_p50_s": percentile(e2e, 0.50),
        "e2e_p95_s": percentile(e2e, 0.95),
        "tpot_p50_s": percentile(tpot, 0.50),
        "tpot_p95_s": percentile(tpot, 0.95),
        "queue_p50_s": percentile(queue, 0.50),
    }


def main():
    a = args()
    model_path = Path(a.model).resolve()
    result_path = Path(a.result_json).resolve()

    cfg = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if cfg.get("internal_version") != a.expected_internal_version:
        raise RuntimeError(f"unexpected internal_version={cfg.get('internal_version')!r}")
    if a.max_model_len > int(cfg.get("max_position_embeddings", 0)):
        raise RuntimeError("requested max_model_len exceeds checkpoint config")

    print("=" * 72, flush=True)
    print(f"ALLaM characterization: TP={a.tp} context={a.max_model_len}", flush=True)
    print("Python:", sys.version.replace("\n", " "), flush=True)
    print("Platform:", platform.platform(), flush=True)
    print("vLLM:", getattr(vllm, "__version__", "unknown"), flush=True)
    print(subprocess.check_output(["nvidia-smi", "-L"], text=True), flush=True)

    init_t0 = time.perf_counter()
    llm = KaggleLLM(
        model=str(model_path),
        tensor_parallel_size=a.tp,
        dtype="float16",
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        trust_remote_code=False,
        max_num_seqs=max(8, max(int(x) for x in a.concurrency_levels.split(","))),
    )
    engine_init_s = time.perf_counter() - init_t0
    tokenizer = llm.upstream.get_tokenizer()

    # Post-engine CUDA provenance.
    gpu_records = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpu_records.append(
            {
                "index": i,
                "name": p.name,
                "compute_capability": f"{p.major}.{p.minor}",
                "total_memory_gib": round(p.total_memory / 1024**3, 3),
            }
        )

    rows = []

    # Warm-up: excluded from summaries.
    rows.append(
        run_batch(
            llm, tokenizer, "warmup", "english",
            prompt_tokens=64, output_tokens=8, concurrency=1, repeat=0
        )
    )

    # Repeated exact-token core workload.
    for rep in range(1, a.repeats + 1):
        for language in ("english", "arabic"):
            rows.append(
                run_batch(
                    llm, tokenizer, "core", language,
                    prompt_tokens=a.core_prompt_tokens,
                    output_tokens=a.core_output_tokens,
                    concurrency=1,
                    repeat=rep,
                )
            )

    # Near-context capacity probe.
    capacity_prompt = a.max_model_len - a.capacity_margin_tokens
    if capacity_prompt + a.capacity_output_tokens > a.max_model_len:
        capacity_prompt = a.max_model_len - a.capacity_output_tokens
    for language in ("english", "arabic"):
        rows.append(
            run_batch(
                llm, tokenizer, "capacity", language,
                prompt_tokens=capacity_prompt,
                output_tokens=a.capacity_output_tokens,
                concurrency=1,
                repeat=1,
            )
        )

    # Offline scheduler concurrency scaling.
    for c in [int(x) for x in a.concurrency_levels.split(",") if x.strip()]:
        language = "english" if c % 2 else "arabic"
        rows.append(
            run_batch(
                llm, tokenizer, "concurrency", language,
                prompt_tokens=a.concurrency_prompt_tokens,
                output_tokens=a.concurrency_output_tokens,
                concurrency=c,
                repeat=1,
            )
        )

    # Normal chat sanity (EOS enabled).
    conversations = [
        [{"role": "user", "content": "Who are you? Answer in one concise sentence."}],
        [{"role": "user", "content": "من أنت؟ أجب بجملة واحدة مختصرة."}],
        [
            {"role": "user", "content": "What is tensor parallelism? One sentence."},
            {"role": "assistant", "content": "It partitions tensor operations across accelerators."},
            {"role": "user", "content": "Why can communication overhead matter? One sentence."},
        ],
        [
            {"role": "user", "content": "ما هو التوازي على مستوى المصفوفات؟ جملة واحدة."},
            {"role": "assistant", "content": "يقسم العمليات الحسابية بين عدة معالجات."},
            {"role": "user", "content": "لماذا قد تكون كلفة الاتصال مهمة؟ جملة واحدة."},
        ],
    ]
    chat_prompts = [
        tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True)
        for x in conversations
    ]
    chat_out = llm.generate(
        chat_prompts,
        SamplingParams(temperature=0.0, max_tokens=64, seed=0),
        use_tqdm=False,
    )
    chat_records = []
    for i, o in enumerate(chat_out):
        cand = o.outputs[0]
        chat_records.append(
            {
                "index": i,
                "prompt_tokens": len(o.prompt_token_ids or []),
                "output_tokens": len(cand.token_ids or []),
                "finish_reason": getattr(cand, "finish_reason", None),
                "text": (cand.text or "").strip(),
            }
        )

    result = {
        "status": "PASS",
        "model_repo": "humain-ai/ALLaM-7B-Instruct-preview",
        "model_commit": a.model_commit,
        "internal_version": cfg.get("internal_version"),
        "sdk": "kaggle-vllm==0.2.0",
        "vllm_distribution_version": getattr(vllm, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "tp": a.tp,
        "max_model_len": a.max_model_len,
        "dtype": "float16",
        "gpu_memory_utilization": a.gpu_memory_utilization,
        "enforce_eager": True,
        "disable_custom_all_reduce": True,
        "worker_multiproc_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
        "engine_init_s": engine_init_s,
        "gpus_visible_to_child": gpu_records,
        "benchmarks": rows,
        "chat_sanity": chat_records,
    }

    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = result_path.with_suffix(result_path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(result_path)
    print("CHARACTERIZATION_PASS", result_path, flush=True)


if __name__ == "__main__":
    main()
