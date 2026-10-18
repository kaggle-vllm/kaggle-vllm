#!/usr/bin/python3.12
"""Offline Kaggle L4x4 preflight and optional local-model inference.

Run with /usr/bin/python3.12 after installing the candidate SDK and pinned
runtime wheels into /usr/local/lib/python3.12/dist-packages. No pip invocation
or download occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="existing local Gemma 4 model directory; runs one TP=4 prompt")
    parser.add_argument("--native-wheel", type=Path, help="optional pinned offline native wheel for archive SHA256 verification")
    parser.add_argument("--prompt", default="Reply with one short sentence: what is tensor parallelism?")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "VLLM_NO_USAGE_STATS"):
        os.environ[key] = "1"
    result: dict[str, object] = {"python": sys.executable, "runtime_mode": "global", "inference": "NOT_RUN"}
    try:
        from kaggle_vllm import KaggleLLM, KaggleSamplingParams
        from kaggle_vllm.global_runtime import validate_global_runtime
        from kaggle_vllm.llm import _load_llm_class
        from kaggle_vllm.profile_v3 import PROFILE_NAME, load_profile
        from kaggle_vllm.runtime import validate_l4x4_profile

        result["runtime"] = validate_global_runtime()
        validate_l4x4_profile()
        result["gpu_topology"] = "PASS: four NVIDIA L4 / SM89"
        _load_llm_class(allow_t4_fallback=False)
        result["vllm_llm_lazy_import"] = "PASS"
        if args.native_wheel is not None:
            actual = sha256(args.native_wheel)
            expected = load_profile().wheel_sha256
            if actual != expected:
                raise ValueError(f"native wheel SHA256 mismatch: expected {expected}, found {actual}")
            result["native_wheel_sha256"] = actual
        if args.model is not None:
            engine = KaggleLLM(args.model, profile=PROFILE_NAME, runtime_mode="global", strict=True, tensor_parallel_size=4)
            output = engine.generate([args.prompt], KaggleSamplingParams(temperature=0, max_tokens=args.max_tokens))
            result["inference"] = "PASS"
            result["output_count"] = len(output)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        result["status"] = "FAILED"
        result["error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
