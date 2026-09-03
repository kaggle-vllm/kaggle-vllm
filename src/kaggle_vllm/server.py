"""Safe argument-array construction for upstream vLLM OpenAI serving."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .runtime import validate_tensor_parallel_size


@dataclass(frozen=True)
class ServerConfig:
    """Supported settings for the validated OpenAI-compatible server workflow."""

    model: str
    served_model_name: str | None = None
    tensor_parallel_size: int = 1
    load_format: str | None = None
    dtype: str = "float16"
    max_model_len: int | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    gpu_memory_utilization: float | None = None
    enforce_eager: bool = True
    disable_custom_all_reduce: bool = True
    model_revision: str | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    seed: int | None = None
    enable_prefix_caching: bool | None = None


def build_server_command(
    config: ServerConfig,
    *,
    executable: str | Path = "vllm",
    extra_args: Sequence[str] = (),
    validate_gpus: bool = True,
) -> list[str]:
    """Return a subprocess-safe upstream ``vllm serve`` argument list."""

    if validate_gpus:
        validate_tensor_parallel_size(config.tensor_parallel_size)
    if not 1 <= config.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if config.max_model_len is not None and config.max_model_len < 1:
        raise ValueError("max_model_len must be positive")
    if config.gpu_memory_utilization is not None and not (
        0.0 < config.gpu_memory_utilization <= 1.0
    ):
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    for name, value in (
        ("max_num_batched_tokens", config.max_num_batched_tokens),
        ("max_num_seqs", config.max_num_seqs),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer when set")
    if config.seed is not None and (
        isinstance(config.seed, bool) or not isinstance(config.seed, int)
    ):
        raise TypeError("seed must be an integer when set")
    if any("\0" in value for value in extra_args):
        raise ValueError("server arguments may not contain NUL bytes")

    command = [str(executable), "serve", config.model]
    if config.served_model_name:
        command.extend(["--served-model-name", config.served_model_name])
    if config.model_revision:
        command.extend(["--revision", config.model_revision])
    if config.load_format:
        command.extend(["--load-format", config.load_format])
    command.extend(
        [
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--tensor-parallel-size",
            str(config.tensor_parallel_size),
            "--dtype",
            config.dtype,
        ]
    )
    if config.max_model_len is not None:
        command.extend(["--max-model-len", str(config.max_model_len)])
    if config.gpu_memory_utilization is not None:
        command.extend(["--gpu-memory-utilization", str(config.gpu_memory_utilization)])
    if config.max_num_batched_tokens is not None:
        command.extend(
            ["--max-num-batched-tokens", str(config.max_num_batched_tokens)]
        )
    if config.max_num_seqs is not None:
        command.extend(["--max-num-seqs", str(config.max_num_seqs)])
    if config.seed is not None:
        command.extend(["--seed", str(config.seed)])
    if config.enable_prefix_caching is not None:
        command.append(
            "--enable-prefix-caching"
            if config.enable_prefix_caching
            else "--no-enable-prefix-caching"
        )
    if config.enforce_eager:
        command.append("--enforce-eager")
    if config.disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")
    command.extend(extra_args)
    return command


def serve(
    config: ServerConfig,
    *,
    executable: str | Path = "vllm",
    extra_args: Sequence[str] = (),
) -> int:
    """Run upstream vLLM in the foreground without invoking a shell."""

    command = build_server_command(config, executable=executable, extra_args=extra_args)
    return subprocess.run(command, check=False).returncode
