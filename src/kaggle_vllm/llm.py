"""Thin, lazy wrapper around upstream :class:`vllm.LLM`."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from .bootstrap import activate_runtime
from .exceptions import ShardedModelError, VLLMNotInstalledError
from .runtime import validate_tensor_parallel_size
from .sharding import ShardedModelInspection, copy_model_metadata, inspect_sharded_model

T4_CONSERVATIVE_DEFAULTS: dict[str, Any] = {
    "dtype": "float16",
    "enforce_eager": True,
    "disable_custom_all_reduce": True,
}


def _load_llm_class() -> type[Any]:
    try:
        module = importlib.import_module("vllm")
        return module.LLM
    except (ImportError, AttributeError, OSError):
        try:
            activated = activate_runtime()
            if activated:
                module = importlib.import_module("vllm")
                return module.LLM
        except (ImportError, AttributeError, OSError, RuntimeError):
            pass
        raise VLLMNotInstalledError(
            "vLLM is unavailable. Run the explicit `kaggle-vllm bootstrap` command "
            "in the compatible Kaggle CPython 3.12 runtime, or activate an existing "
            "runtime manifest; diagnostics work without vLLM."
        ) from None


class KaggleLLM:
    """Construct upstream vLLM with validated TP size and conservative T4 defaults.

    All extra keyword arguments are forwarded to ``vllm.LLM``. Explicit user
    values override the defaults documented for the validated Kaggle T4 profile.
    """

    def __init__(
        self,
        model: str | Path,
        tensor_parallel_size: int = 1,
        **vllm_kwargs: Any,
    ) -> None:
        validate_tensor_parallel_size(tensor_parallel_size)
        options = dict(T4_CONSERVATIVE_DEFAULTS)
        options.update(vllm_kwargs)
        options["model"] = str(model)
        options["tensor_parallel_size"] = tensor_parallel_size

        self.model = str(model)
        self.tensor_parallel_size = tensor_parallel_size
        self.engine_args = options.copy()
        self._llm = _load_llm_class()(**options)

    @property
    def upstream(self) -> Any:
        """Return the wrapped upstream vLLM object for advanced workflows."""

        return self._llm

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate generation to upstream ``vllm.LLM.generate``."""

        return self._llm.generate(*args, **kwargs)

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
