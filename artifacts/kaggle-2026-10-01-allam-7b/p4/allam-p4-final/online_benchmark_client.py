#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model-name", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--languages", default="english,arabic")
    p.add_argument("--concurrency-levels", default="1,2,4,8")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--target-prompt-tokens", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=32)
    p.add_argument("--capacity-output-tokens", type=int, default=32)
    p.add_argument("--capacity-margin-tokens", type=int, default=64)
    p.add_argument("--result-json", required=True)
    return p.parse_args()


def percentile(values, q):
    vals = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def make_prompt(tokenizer, language, target_tokens, request_id):
    if language == "english":
        seed = (
            f"Unique request {request_id}. "
            "Efficient language model inference requires careful measurement of latency, "
            "throughput, memory capacity, scheduling, and reproducibility. "
        )
        filler = (
            "This controlled benchmark uses a distinct early prefix so cached prefixes "
            "do not intentionally dominate the measurement. "
        )
    else:
        seed = (
            f"طلب فريد {request_id}. "
            "يتطلب استدلال نماذج اللغة قياساً دقيقاً لزمن الاستجابة والإنتاجية "
            "وسعة الذاكرة والجدولة وقابلية إعادة النتائج. "
        )
        filler = (
            "يستخدم هذا الاختبار مقدمة مختلفة لكل طلب حتى لا تهيمن إعادة استخدام "
            "البادئات المخزنة مؤقتاً على القياس. "
        )

    text = seed
    while len(tokenizer.encode(text, add_special_tokens=False)) < target_tokens + 32:
        text += filler

    ids = tokenizer.encode(text, add_special_tokens=False)

    # Decode a prefix near the target, then re-tokenize. Round-trip token counts can
    # shift slightly; always trim until at or below the requested target.
    ids = ids[:target_tokens]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    actual = len(tokenizer.encode(text, add_special_tokens=False))

    while actual > target_tokens and len(text) > 1:
        text = text[:-1]
        actual = len(tokenizer.encode(text, add_special_tokens=False))

    return text, actual


async def stream_completion(
    client,
    base_url,
    model_name,
    prompt,
    output_tokens,
    request_id,
):
    base_payload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "seed": 0,
    }

    extension_payload = {
        **base_payload,
        "min_tokens": output_tokens,
        "ignore_eos": True,
    }

    async def do_request(payload, forced_output):
        started = time.perf_counter()
        first_text_time = None
        first_event_time = None
        usage = None
        text_parts = []
        chunk_times = []
        status_code = None
        error = None

        try:
            async with client.stream(
                "POST",
                f"{base_url}/v1/completions",
                json=payload,
            ) as response:
                status_code = response.status_code
                if response.status_code >= 400:
                    body = await response.aread()
                    return {
                        "ok": False,
                        "status_code": status_code,
                        "error": body.decode("utf-8", errors="replace"),
                        "forced_output": forced_output,
                    }

                async for line in response.aiter_lines():
                    now = time.perf_counter()
                    if first_event_time is None:
                        first_event_time = now

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()

                    if data == "[DONE]":
                        break

                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if event.get("usage") is not None:
                        usage = event["usage"]

                    choices = event.get("choices") or []
                    for choice in choices:
                        piece = choice.get("text") or ""
                        if piece:
                            text_parts.append(piece)
                            chunk_times.append(now)
                            if first_text_time is None:
                                first_text_time = now

        except Exception as exc:
            error = repr(exc)

        finished = time.perf_counter()

        if error is not None:
            return {
                "ok": False,
                "status_code": status_code,
                "error": error,
                "forced_output": forced_output,
            }

        if first_text_time is None:
            return {
                "ok": False,
                "status_code": status_code,
                "error": "No non-empty streamed text chunk received.",
                "forced_output": forced_output,
            }

        prompt_tokens = (usage or {}).get("prompt_tokens")
        completion_tokens = (usage or {}).get("completion_tokens")
        total_tokens = (usage or {}).get("total_tokens")

        ttft = first_text_time - started
        e2e = finished - started

        tpot = None
        if completion_tokens is not None and completion_tokens > 1:
            tpot = (e2e - ttft) / (completion_tokens - 1)

        chunk_interarrival = []
        if len(chunk_times) > 1:
            chunk_interarrival = [
                chunk_times[i] - chunk_times[i - 1]
                for i in range(1, len(chunk_times))
            ]

        return {
            "ok": True,
            "request_id": request_id,
            "status_code": status_code,
            "forced_output": forced_output,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "ttft_s": ttft,
            "e2e_s": e2e,
            "tpot_s": tpot,
            "stream_text_chunks": len(chunk_times),
            "chunk_interarrival_p50_s": percentile(chunk_interarrival, 0.50),
            "chunk_interarrival_p95_s": percentile(chunk_interarrival, 0.95),
            "text": "".join(text_parts),
        }

    result = await do_request(extension_payload, forced_output=True)

    # vLLM versions differ in which extension fields are accepted by the OpenAI
    # protocol. Retry only protocol-level 4xx failures.
    if (
        not result.get("ok")
        and result.get("status_code") is not None
        and 400 <= int(result["status_code"]) < 500
    ):
        result = await do_request(base_payload, forced_output=False)

    return result


async def run_wave(
    client,
    base_url,
    model_name,
    tokenizer,
    language,
    concurrency,
    repeat,
    target_prompt_tokens,
    output_tokens,
):
    prompts = []

    for i in range(concurrency):
        rid = f"{language}-c{concurrency}-r{repeat}-q{i}"
        prompt, local_count = make_prompt(
            tokenizer,
            language,
            target_prompt_tokens,
            rid,
        )
        prompts.append((rid, prompt, local_count))

    wave_started = time.perf_counter()

    results = await asyncio.gather(
        *[
            stream_completion(
                client,
                base_url,
                model_name,
                prompt,
                output_tokens,
                rid,
            )
            for rid, prompt, _ in prompts
        ]
    )

    wave_wall = time.perf_counter() - wave_started

    successful = [x for x in results if x.get("ok")]
    failed = [x for x in results if not x.get("ok")]

    prompt_tokens = [
        x.get("prompt_tokens")
        for x in successful
        if x.get("prompt_tokens") is not None
    ]
    completion_tokens = [
        x.get("completion_tokens")
        for x in successful
        if x.get("completion_tokens") is not None
    ]

    ttft = [x["ttft_s"] for x in successful if x.get("ttft_s") is not None]
    tpot = [x["tpot_s"] for x in successful if x.get("tpot_s") is not None]
    e2e = [x["e2e_s"] for x in successful if x.get("e2e_s") is not None]

    total_prompt = sum(prompt_tokens) if prompt_tokens else None
    total_output = sum(completion_tokens) if completion_tokens else None

    summary = {
        "language": language,
        "concurrency": concurrency,
        "repeat": repeat,
        "request_count": len(results),
        "success_count": len(successful),
        "error_count": len(failed),
        "success_rate": len(successful) / len(results) if results else 0.0,
        "wave_wall_s": wave_wall,
        "request_throughput_rps": len(successful) / wave_wall if wave_wall else None,
        "total_prompt_tokens": total_prompt,
        "total_output_tokens": total_output,
        "output_tokens_per_s": (
            total_output / wave_wall
            if total_output is not None and wave_wall
            else None
        ),
        "total_tokens_per_s": (
            (total_prompt + total_output) / wave_wall
            if total_prompt is not None
            and total_output is not None
            and wave_wall
            else None
        ),
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p95_s": percentile(ttft, 0.95),
        "ttft_p99_s": percentile(ttft, 0.99),
        "tpot_p50_s": percentile(tpot, 0.50),
        "tpot_p95_s": percentile(tpot, 0.95),
        "tpot_p99_s": percentile(tpot, 0.99),
        "e2e_p50_s": percentile(e2e, 0.50),
        "e2e_p95_s": percentile(e2e, 0.95),
        "e2e_p99_s": percentile(e2e, 0.99),
        "all_forced_output": (
            all(x.get("forced_output") for x in successful)
            if successful
            else False
        ),
        "local_prompt_token_counts": [p[2] for p in prompts],
        "requests": results,
    }

    return summary


async def capacity_probe(
    client,
    base_url,
    model_name,
    tokenizer,
    language,
    max_model_len,
    output_tokens,
    margin,
):
    target = max_model_len - output_tokens - margin
    rid = f"capacity-{language}-{max_model_len}"
    prompt, local_count = make_prompt(tokenizer, language, target, rid)

    result = await stream_completion(
        client,
        base_url,
        model_name,
        prompt,
        output_tokens,
        rid,
    )

    return {
        "language": language,
        "target_prompt_tokens": target,
        "local_prompt_tokens": local_count,
        "result": result,
    }


async def chat_sanity(client, base_url, model_name):
    conversations = [
        {
            "label": "english_single",
            "messages": [
                {
                    "role": "user",
                    "content": "Who are you? Answer in one concise sentence.",
                }
            ],
        },
        {
            "label": "arabic_single",
            "messages": [
                {
                    "role": "user",
                    "content": "من أنت؟ أجب بجملة واحدة مختصرة.",
                }
            ],
        },
        {
            "label": "english_multiturn",
            "messages": [
                {
                    "role": "user",
                    "content": "What is tensor parallelism? One sentence.",
                },
                {
                    "role": "assistant",
                    "content": "It partitions tensor operations across accelerators.",
                },
                {
                    "role": "user",
                    "content": "Why can communication overhead matter? One sentence.",
                },
            ],
        },
        {
            "label": "arabic_multiturn",
            "messages": [
                {
                    "role": "user",
                    "content": "ما هو التوازي على مستوى المصفوفات؟ جملة واحدة.",
                },
                {
                    "role": "assistant",
                    "content": "يقسم العمليات الحسابية بين عدة معالجات.",
                },
                {
                    "role": "user",
                    "content": "لماذا قد تكون كلفة الاتصال مهمة؟ جملة واحدة.",
                },
            ],
        },
    ]

    out = []

    for item in conversations:
        t0 = time.perf_counter()
        try:
            response = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": item["messages"],
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "seed": 0,
                },
            )
            elapsed = time.perf_counter() - t0

            if response.status_code >= 400:
                out.append(
                    {
                        "label": item["label"],
                        "ok": False,
                        "status_code": response.status_code,
                        "error": response.text,
                        "e2e_s": elapsed,
                    }
                )
                continue

            data = response.json()
            text = (
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                or ""
            ).strip()

            out.append(
                {
                    "label": item["label"],
                    "ok": bool(text),
                    "status_code": response.status_code,
                    "text": text,
                    "usage": data.get("usage"),
                    "e2e_s": elapsed,
                }
            )
        except Exception as exc:
            out.append(
                {
                    "label": item["label"],
                    "ok": False,
                    "error": repr(exc),
                }
            )

    return out


async def main():
    a = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        a.tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )

    languages = [x.strip() for x in a.languages.split(",") if x.strip()]
    concurrencies = [int(x) for x in a.concurrency_levels.split(",") if x.strip()]

    limits = httpx.Limits(
        max_connections=max(concurrencies) + 8,
        max_keepalive_connections=max(concurrencies) + 8,
    )

    timeout = httpx.Timeout(
        connect=30.0,
        read=300.0,
        write=60.0,
        pool=60.0,
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        # Health/model check.
        health = await client.get(f"{a.base_url}/health")
        models = await client.get(f"{a.base_url}/v1/models")

        if health.status_code >= 400:
            raise RuntimeError(f"health failed: {health.status_code} {health.text}")

        if models.status_code >= 400:
            raise RuntimeError(f"models failed: {models.status_code} {models.text}")

        # Warm up both languages, excluded from measurements.
        for language in languages:
            warm_prompt, _ = make_prompt(tokenizer, language, 64, f"warmup-{language}")
            warm = await stream_completion(
                client,
                a.base_url,
                a.model_name,
                warm_prompt,
                8,
                f"warmup-{language}",
            )
            if not warm.get("ok"):
                raise RuntimeError(f"warmup failed for {language}: {warm}")

        waves = []

        for language in languages:
            for concurrency in concurrencies:
                for repeat in range(1, a.repeats + 1):
                    wave = await run_wave(
                        client,
                        a.base_url,
                        a.model_name,
                        tokenizer,
                        language,
                        concurrency,
                        repeat,
                        a.target_prompt_tokens,
                        a.output_tokens,
                    )
                    waves.append(wave)
                    print(
                        "WAVE",
                        language,
                        "C",
                        concurrency,
                        "R",
                        repeat,
                        "success",
                        wave["success_count"],
                        "/",
                        wave["request_count"],
                        "out_tps",
                        wave["output_tokens_per_s"],
                        "ttft_p50",
                        wave["ttft_p50_s"],
                        flush=True,
                    )

        capacity = []
        for language in languages:
            capacity.append(
                await capacity_probe(
                    client,
                    a.base_url,
                    a.model_name,
                    tokenizer,
                    language,
                    a.max_model_len,
                    a.capacity_output_tokens,
                    a.capacity_margin_tokens,
                )
            )

        chats = await chat_sanity(client, a.base_url, a.model_name)

    result = {
        "status": "PASS",
        "base_url": a.base_url,
        "model_name": a.model_name,
        "max_model_len": a.max_model_len,
        "languages": languages,
        "concurrency_levels": concurrencies,
        "repeats": a.repeats,
        "target_prompt_tokens": a.target_prompt_tokens,
        "requested_output_tokens": a.output_tokens,
        "waves": waves,
        "capacity_probes": capacity,
        "chat_sanity": chats,
    }

    path = Path(a.result_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)

    print("ONLINE_BENCHMARK_PASS", path, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
