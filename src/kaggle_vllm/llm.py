"""Thin, lazy wrapper around upstream :class:`vllm.LLM`."""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path
from typing import Any

from .l4_bootstrap import activate_runtime
from .bootstrap import activate_runtime as activate_t4_runtime
from .environment import collect, detect_kaggle
from .exceptions import RuntimeValidationError, ShardedModelError, VLLMLazyImportError, VLLMNotInstalledError
from .runtime import all_sm75, all_tesla_t4, validate_l4x4_profile, validate_tensor_parallel_size
from .profile_v3 import PROFILE_NAME as L4_PROFILE
from .profiles import DEFAULT_PROFILE as T4_PROFILE
from .sampling import KaggleSamplingParams
from .sharding import ShardedModelInspection, copy_model_metadata, inspect_sharded_model

L4_COMPATIBILITY_DEFAULTS: dict[str, Any] = {
    "enforce_eager": True,
    "disable_custom_all_reduce": True,
    "trust_remote_code": False,
}
T4_CONSERVATIVE_DEFAULTS: dict[str, Any] = {
    "dtype": "float16",
    "enforce_eager": True,
    "disable_custom_all_reduce": True,
}


def _detect_kaggle_profile_without_torch() -> str:
    """Use driver inventory before activation so base Torch is never preloaded."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeValidationError(f"cannot detect Kaggle GPU profile; pass profile explicitly: {error}") from error
    devices = [tuple(part.strip() for part in line.split(",", 1)) for line in result.stdout.splitlines() if line.strip()]
    if devices == [("NVIDIA L4", "8.9")] * 4:
        return L4_PROFILE
    if devices == [("Tesla T4", "7.5")] * 2 or devices == [("NVIDIA Tesla T4", "7.5")] * 2:
        return T4_PROFILE
    raise RuntimeValidationError(f"hardware profile is ambiguous; pass profile explicitly: {devices}")


def _load_vllm_module(profile: str | None = None, runtime_mode: str = "managed") -> Any:
    if runtime_mode not in ("managed", "global"):
        raise RuntimeValidationError(f"unknown runtime_mode: {runtime_mode}")
    if runtime_mode != "global" and profile != T4_PROFILE and not activate_runtime():
        raise VLLMNotInstalledError(
            "vLLM runtime is not staged; run `kaggle-vllm bootstrap --strict`"
        )
    try:
        return importlib.import_module("vllm")
    except Exception as error:
        if isinstance(error, ModuleNotFoundError) and error.name == "vllm":
            raise VLLMNotInstalledError(f"vLLM is absent in {runtime_mode} mode: {error}") from error
        raise VLLMLazyImportError(f"vLLM module import failed in {runtime_mode} mode: {type(error).__name__}: {error}") from error


def _load_llm_class(*, allow_t4_fallback: bool = True) -> type[Any]:
    try:
        return importlib.import_module("vllm").LLM
    except Exception as error:
        if allow_t4_fallback and isinstance(error, ModuleNotFoundError) and error.name == "vllm":
            try:
                if activate_t4_runtime():
                    return importlib.import_module("vllm").LLM
            except Exception as retry_error:
                raise VLLMLazyImportError(
                    f"vLLM.LLM failed after T4 runtime activation: {type(retry_error).__name__}: {retry_error}"
                ) from retry_error
        if isinstance(error, ModuleNotFoundError) and error.name == "vllm":
            raise VLLMNotInstalledError(f"vLLM package is absent; bootstrap the selected managed runtime or install the pinned global wheel: {error}") from error
        raise VLLMLazyImportError(f"vLLM.LLM lazy import failed: {type(error).__name__}: {error}; check the named dependency or ABI") from error


class KaggleLLM:
    """Construct upstream vLLM with P31 L4x4 compatibility defaults.

    All extra keyword arguments are forwarded to ``vllm.LLM``. Explicit user
    values override the documented P31 compatibility defaults.
    """

    def __init__(
        self,
        model: str | Path,
        tensor_parallel_size: int | None = None,
        strict: bool | None = None,
        profile: str | None = None,
        runtime_mode: str = "managed",
        **vllm_kwargs: Any,
    ) -> None:
        strict_mode = detect_kaggle() if strict is None else strict
        if profile is None and strict_mode:
            profile = _detect_kaggle_profile_without_torch()
        # Preserve the v0.2.0 local default. Kaggle hardware detection above
        # selects L4 only when the actual four-L4 inventory is observed.
        implicit_t4 = profile is None
        profile = profile or T4_PROFILE
        if profile not in (L4_PROFILE, T4_PROFILE):
            raise RuntimeValidationError(f"unknown profile: {profile}")
        if runtime_mode not in ("managed", "global"):
            raise RuntimeValidationError(f"unknown runtime_mode: {runtime_mode}")
        if runtime_mode == "global" and profile != L4_PROFILE:
            raise RuntimeValidationError("global CPython 3.12 mode supports only the L4x4 profile")
        if profile == T4_PROFILE:
            from . import l4_bootstrap
            if l4_bootstrap._ACTIVE_L4_TARGET is not None:
                raise RuntimeValidationError("L4 runtime is active in this process; restart before selecting T4")
        if profile == L4_PROFILE and runtime_mode == "global":
            from .global_runtime import validate_global_runtime
            validate_global_runtime()
        elif profile == L4_PROFILE and not activate_runtime():
            raise VLLMNotInstalledError("L4 runtime is not staged; run `kaggle-vllm bootstrap --profile kaggle-l4x4-cu129 --offline`")
        if tensor_parallel_size is None:
            tensor_parallel_size = (1 if implicit_t4 else 2) if profile == T4_PROFILE else 4
        if strict_mode or runtime_mode == "global":
            if profile == L4_PROFILE:
                if runtime_mode == "managed":
                    activate_runtime()
                validate_l4x4_profile()
                validate_tensor_parallel_size(tensor_parallel_size, gpu_count=4)
                if tensor_parallel_size != 4:
                    raise RuntimeValidationError("strict L4x4 profile requires tensor_parallel_size=4")
            else:
                snapshot = collect()
                if snapshot.gpu_count != 2 or not all_tesla_t4(snapshot.gpus) or not all_sm75(snapshot.gpus) or tensor_parallel_size != 2:
                    raise RuntimeValidationError("strict T4 profile requires exactly 2 Tesla T4 / SM75 GPUs and TP=2")
        elif not isinstance(tensor_parallel_size, int) or isinstance(tensor_parallel_size, bool) or tensor_parallel_size < 1:
            validate_tensor_parallel_size(tensor_parallel_size, gpu_count=0)
        options = dict(T4_CONSERVATIVE_DEFAULTS if profile == T4_PROFILE else L4_COMPATIBILITY_DEFAULTS)
        options.update(vllm_kwargs)
        if runtime_mode == "global":
            try:
                local_model = Path(model).expanduser().resolve(strict=True)
            except (OSError, TypeError, ValueError) as error:
                raise RuntimeValidationError(f"global mode requires a local model directory: {model}") from error
            if not local_model.is_dir() or not (local_model / "config.json").is_file():
                raise RuntimeValidationError("global mode requires a local model directory with config.json")
            tokenizer = options.get("tokenizer")
            if tokenizer is not None and (not Path(tokenizer).expanduser().is_dir() or not (Path(tokenizer).expanduser() / "tokenizer_config.json").is_file()):
                raise RuntimeValidationError("global mode tokenizer must be a local directory with tokenizer_config.json")
            if options.get("trust_remote_code") is not False:
                raise RuntimeValidationError("global mode requires trust_remote_code=False")
            model = local_model
        options["model"] = str(model)
        options["tensor_parallel_size"] = tensor_parallel_size

        self.model = str(model)
        self.profile = profile
        self.runtime_mode = runtime_mode
        self.tensor_parallel_size = tensor_parallel_size
        self.engine_args = options.copy()
        self._llm = _load_llm_class(allow_t4_fallback=profile == T4_PROFILE)(**options)

    @property
    def upstream(self) -> Any:
        """Return the wrapped upstream vLLM object for advanced workflows."""

        return self._llm

    def generate(self, prompts: Any, sampling_params: Any = None, **kwargs: Any) -> Any:
        """Delegate generation to upstream ``vllm.LLM.generate``."""
        if isinstance(sampling_params, KaggleSamplingParams):
            sampling_params = sampling_params.to_upstream(T4_PROFILE) if self.profile == T4_PROFILE else sampling_params.to_upstream(runtime_mode=self.runtime_mode)
        if sampling_params is None:
            return self._llm.generate(prompts, **kwargs)
        return self._llm.generate(prompts, sampling_params, **kwargs)

    def generate_text(self, prompts: Any, *, temperature: float = 0.0, max_tokens: int = 64, **sampling_kwargs: Any) -> Any:
        """Generate with SDK sampling options; returns upstream request outputs."""
        return self.generate(prompts, KaggleSamplingParams(temperature=temperature, max_tokens=max_tokens, **sampling_kwargs))

    def chat(self, messages: list[dict[str, str]], *, local_files_only: bool = True, sampling_params: KaggleSamplingParams | Any | None = None, **kwargs: Any) -> Any:
        """Render a model chat template with AutoProcessor, then AutoTokenizer fallback."""
        if self.runtime_mode == "global" and not local_files_only:
            raise RuntimeValidationError("global mode requires local_files_only=True")
        from transformers import AutoProcessor, AutoTokenizer
        errors: list[Exception] = []
        for factory in (AutoProcessor, AutoTokenizer):
            try:
                renderer = factory.from_pretrained(self.model, local_files_only=local_files_only, trust_remote_code=False)
                prompt = renderer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("chat template returned an empty or non-text prompt")
                break
            except (OSError, ValueError, TypeError, AttributeError) as error:
                errors.append(error)
        else:
            raise RuntimeError(f"model chat template unavailable: {errors[-1]}")
        return self.generate([prompt], sampling_params, **kwargs)

    def effective_attention_backend(self) -> str | None:
        """Best-effort model-specific diagnostic; selection belongs to upstream vLLM."""
        try:
            config = self._llm.llm_engine.vllm_config
            backend = getattr(config, "attention_backend", None)
            return str(backend) if backend is not None else None
        except AttributeError:
            return None

    def save_sharded_model(
        self,
        path: str | Path,
        *,
        max_size: int | None = 2 * 1024**3,
        metadata_source: str | Path | None = None,
    ) -> ShardedModelInspection:
        """Save a vLLM-native persistent TP-aware ``sharded_state`` checkpoint.

        The destination must be absent or empty. Model/tokenizer metadata is copied
        when ``metadata_source`` (or the original model argument) is a local folder.
        """

        expanded = Path(path).expanduser()
        lexical = Path(os.path.abspath(expanded))
        destination = expanded.resolve(strict=False)
        if lexical != destination:
            raise ShardedModelError(
                "refusing sharded-state destination that traverses a symlink: "
                f"{lexical} -> {destination}"
            )
        if destination.exists() and not destination.is_dir():
            raise ShardedModelError(
                f"destination exists and is not a directory: {destination}"
            )
        if destination.exists() and any(destination.iterdir()):
            raise ShardedModelError(
                f"refusing to overwrite non-empty directory: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)

        try:
            save = self._llm.llm_engine.engine_core.save_sharded_state
        except AttributeError as error:
            raise ShardedModelError(
                "this vLLM engine does not expose engine_core.save_sharded_state"
            ) from error

        arguments: dict[str, Any] = {"path": str(destination)}
        if max_size is not None:
            arguments["max_size"] = max_size
        save(**arguments)
        copy_model_metadata(metadata_source or self.model, destination)
        return inspect_sharded_model(destination)

    def __getattr__(self, name: str) -> Any:
        """Delegate non-wrapper attributes to upstream vLLM."""

        return getattr(self._llm, name)
