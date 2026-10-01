#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# Environment MUST be configured before importing vLLM
# ============================================================

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

# vLLM normally defaults to fork for worker multiprocessing.
#
# This experiment explicitly uses spawn to avoid inheriting a
# pre-existing CUDA state into TP workers.
os.environ.setdefault(
    "VLLM_WORKER_MULTIPROC_METHOD",
    "spawn",
)

# If NCCL initialization fails again, preserve useful details.
os.environ.setdefault(
    "NCCL_DEBUG",
    "INFO",
)

# Helpful NCCL provenance.
os.environ.setdefault(
    "NCCL_DEBUG_SUBSYS",
    "INIT,ENV,GRAPH",
)


# ============================================================
# Activate kaggle-vllm runtime BEFORE importing vLLM
# ============================================================

from kaggle_vllm import (
    KaggleLLM,
    activate_runtime,
)

manifest = os.environ.get(
    "KAGGLE_VLLM_MANIFEST"
)

if not manifest:
    raise RuntimeError(
        "KAGGLE_VLLM_MANIFEST is not set."
    )

if not activate_runtime(manifest):
    raise RuntimeError(
        "Failed to activate kaggle-vllm runtime:\n"
        f"{manifest}"
    )


# Import upstream vLLM only after runtime activation.
import vllm

from vllm import SamplingParams


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--model-commit",
        required=True,
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--result-json",
        required=True,
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    model_path = Path(
        args.model
    ).resolve()

    result_path = Path(
        args.result_json
    ).resolve()


    # --------------------------------------------------------
    # Model snapshot validation — CPU/filesystem only
    # --------------------------------------------------------

    if not model_path.is_dir():
        raise RuntimeError(
            f"Model directory does not exist:\n"
            f"{model_path}"
        )

    config_path = (
        model_path / "config.json"
    )

    if not config_path.is_file():
        raise RuntimeError(
            f"Missing config.json:\n"
            f"{config_path}"
        )

    config = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        config.get("model_type")
        != "llama"
    ):
        raise RuntimeError(
            "Unexpected model_type: "
            f"{config.get('model_type')!r}"
        )

    if (
        config.get("internal_version")
        != "7b-alpha-v2.33.0.30"
    ):
        raise RuntimeError(
            "Unexpected ALLaM internal_version: "
            f"{config.get('internal_version')!r}"
        )


    # --------------------------------------------------------
    # Hardware provenance WITHOUT torch.cuda initialization
    # --------------------------------------------------------

    gpu_list = subprocess.check_output(
        ["nvidia-smi", "-L"],
        text=True,
    ).strip()

    print(
        "============================================================",
        flush=True,
    )

    print(
        "PRE-vLLM ENVIRONMENT",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )

    print(
        "Python:",
        sys.version.replace("\n", " "),
        flush=True,
    )

    print(
        "Platform:",
        platform.platform(),
        flush=True,
    )

    print(
        "vLLM distribution:",
        getattr(
            vllm,
            "__version__",
            "unknown",
        ),
        flush=True,
    )

    print(
        "VLLM_WORKER_MULTIPROC_METHOD:",
        os.environ.get(
            "VLLM_WORKER_MULTIPROC_METHOD"
        ),
        flush=True,
    )

    print(
        "NCCL_DEBUG:",
        os.environ.get(
            "NCCL_DEBUG"
        ),
        flush=True,
    )

    print(
        "GPUs:",
        flush=True,
    )

    print(
        gpu_list,
        flush=True,
    )

    if gpu_list.count("Tesla T4") != 2:
        raise RuntimeError(
            "Expected exactly two Tesla T4 GPUs.\n"
            + gpu_list
        )


    # ========================================================
    # IMPORTANT
    #
    # Do NOT perform:
    #
    #   torch.cuda.is_available()
    #   torch.cuda.device_count()
    #   torch.cuda.get_device_properties()
    #   torch.cuda.get_device_name()
    #
    # before this point.
    #
    # Let vLLM own CUDA initialization for its TP workers.
    # ========================================================

    print(
        "\n============================================================",
        flush=True,
    )

    print(
        "INITIALIZING KaggleLLM TP=2",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )


    engine_started = (
        time.perf_counter()
    )

    llm = KaggleLLM(

        model=str(
            model_path
        ),

        # Actual two-GPU tensor parallelism.
        tensor_parallel_size=2,

        # Tesla T4 / SM75 compatibility.
        dtype="float16",

        # First compatibility run.
        max_model_len=
            args.max_model_len,

        gpu_memory_utilization=
            args.gpu_memory_utilization,

        # Validated kaggle-vllm T4 settings.
        enforce_eager=True,

        disable_custom_all_reduce=True,

        trust_remote_code=False,

        max_num_seqs=4,
    )


    engine_init_seconds = (
        time.perf_counter()
        - engine_started
    )


    print(
        "\nKaggleLLM TP=2 engine initialized.",
        flush=True,
    )

    print(
        "Engine initialization seconds:",
        round(
            engine_init_seconds,
            4,
        ),
        flush=True,
    )


    # --------------------------------------------------------
    # It is now safe to collect Torch/CUDA provenance because
    # the vLLM TP workers have already been established.
    # --------------------------------------------------------

    import torch

    torch_info = {
        "version":
            torch.__version__,

        "cuda":
            torch.version.cuda,

        "cuda_available":
            torch.cuda.is_available(),

        "visible_gpu_count":
            torch.cuda.device_count(),

        "gpus": [],
    }

    for index in range(
        torch.cuda.device_count()
    ):

        props = (
            torch.cuda.get_device_properties(
                index
            )
        )

        torch_info["gpus"].append(
            {
                "index":
                    index,

                "name":
                    props.name,

                "compute_capability":
                    f"{props.major}."
                    f"{props.minor}",

                "total_memory_gib":
                    round(
                        props.total_memory
                        / (1024 ** 3),
                        2,
                    ),
            }
        )


    print(
        "\nPost-engine CUDA provenance:",
        flush=True,
    )

    print(
        json.dumps(
            torch_info,
            indent=2,
        ),
        flush=True,
    )


    if (
        torch_info[
            "visible_gpu_count"
        ]
        != 2
    ):
        raise RuntimeError(
            "vLLM initialized, but the "
            "frontend does not see two GPUs."
        )


    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = (
        llm.upstream.get_tokenizer()
    )

    print(
        "\nTokenizer:",
        tokenizer.__class__.__name__,
        flush=True,
    )


    # --------------------------------------------------------
    # English + Arabic prompts using checkpoint chat template
    # --------------------------------------------------------

    conversations = [

        [
            {
                "role": "user",
                "content":
                    "Who are you? "
                    "Answer in one concise sentence.",
            }
        ],

        [
            {
                "role": "user",
                "content":
                    "من أنت؟ "
                    "أجب بجملة واحدة مختصرة.",
            }
        ],
    ]


    prompts = []

    for messages in conversations:

        prompt = (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        if not prompt:
            raise RuntimeError(
                "ALLaM chat template returned "
                "an empty prompt."
            )

        prompts.append(
            prompt
        )


    for index, prompt in enumerate(
        prompts
    ):

        print(
            f"\nRendered prompt {index}:",
            repr(prompt),
            flush=True,
        )


    # --------------------------------------------------------
    # Deterministic generation
    # --------------------------------------------------------

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
    )


    generation_started = (
        time.perf_counter()
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
    )

    generation_seconds = (
        time.perf_counter()
        - generation_started
    )


    if len(outputs) != 2:
        raise RuntimeError(
            "Expected two vLLM outputs, got "
            f"{len(outputs)}."
        )


    records = []

    total_prompt_tokens = 0
    total_output_tokens = 0


    for index, request_output in enumerate(
        outputs
    ):

        if not request_output.outputs:
            raise RuntimeError(
                f"Request {index} returned "
                "no candidates."
            )

        candidate = (
            request_output.outputs[0]
        )

        generated_text = (
            candidate.text or ""
        ).strip()

        if not generated_text:
            raise RuntimeError(
                f"Request {index} returned "
                "empty generated text."
            )

        prompt_token_ids = (
            request_output.prompt_token_ids
            or []
        )

        output_token_ids = (
            candidate.token_ids
            or []
        )

        prompt_tokens = len(
            prompt_token_ids
        )

        output_tokens = len(
            output_token_ids
        )

        total_prompt_tokens += (
            prompt_tokens
        )

        total_output_tokens += (
            output_tokens
        )

        record = {
            "request_index":
                index,

            "prompt_tokens":
                prompt_tokens,

            "output_tokens":
                output_tokens,

            "finish_reason":
                getattr(
                    candidate,
                    "finish_reason",
                    None,
                ),

            "stop_reason":
                getattr(
                    candidate,
                    "stop_reason",
                    None,
                ),

            "text":
                generated_text,
        }

        records.append(
            record
        )

        print(
            f"\n=== REQUEST {index} ===",
            flush=True,
        )

        print(
            "Prompt tokens:",
            prompt_tokens,
            flush=True,
        )

        print(
            "Output tokens:",
            output_tokens,
            flush=True,
        )

        print(
            "Finish reason:",
            record["finish_reason"],
            flush=True,
        )

        print(
            "Response:",
            flush=True,
        )

        print(
            generated_text,
            flush=True,
        )


    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    result = {

        "status":
            "PASS",

        "sdk":
            "kaggle-vllm==0.2.0",

        "model_repo":
            "humain-ai/"
            "ALLaM-7B-Instruct-preview",

        "model_path":
            str(model_path),

        "model_commit":
            args.model_commit,

        "internal_version":
            config.get(
                "internal_version"
            ),

        "tensor_parallel_size":
            2,

        "dtype":
            "float16",

        "max_model_len":
            args.max_model_len,

        "gpu_memory_utilization":
            args.gpu_memory_utilization,

        "enforce_eager":
            True,

        "disable_custom_all_reduce":
            True,

        "vllm_worker_multiproc_method":
            os.environ.get(
                "VLLM_WORKER_MULTIPROC_METHOD"
            ),

        "vllm_distribution_version":
            getattr(
                vllm,
                "__version__",
                "unknown",
            ),

        "torch":
            torch_info,

        "engine_init_seconds":
            round(
                engine_init_seconds,
                4,
            ),

        "generation_wall_seconds":
            round(
                generation_seconds,
                4,
            ),

        "total_prompt_tokens":
            total_prompt_tokens,

        "total_output_tokens":
            total_output_tokens,

        "outputs":
            records,
    }


    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    temp_path = (
        result_path.with_suffix(
            result_path.suffix + ".tmp"
        )
    )

    temp_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temp_path.replace(
        result_path
    )


    print(
        "\n============================================================",
        flush=True,
    )

    print(
        "ALLaM TP=2 INFERENCE: PASS",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )

    print(
        "Result:",
        result_path,
        flush=True,
    )


if __name__ == "__main__":
    main()
