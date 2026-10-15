"""Lazy facade for upstream vLLM sampling parameters."""
from __future__ import annotations

from typing import Any

from .exceptions import VLLMNotInstalledError


class KaggleSamplingParams:
    """Store options without importing CUDA packages during SDK import."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)

    def to_upstream(self, profile: str | None = None) -> Any:
        from .llm import _load_vllm_module
        module = _load_vllm_module(profile) if profile is not None else _load_vllm_module()
        return module.SamplingParams(**self.kwargs)

    def __repr__(self) -> str:
        return f"KaggleSamplingParams({self.kwargs!r})"
