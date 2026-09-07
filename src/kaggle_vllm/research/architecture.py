"""Explicit model architecture metadata with no guessed model registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ResearchEvidenceError


@dataclass(frozen=True)
class ArchitectureMetadata:
    """Caller-supplied, source-attributed architecture values."""

    model_id: str
    revision: str
    architecture: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    activation_dtype_bytes: int
    source: str


def _positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ResearchEvidenceError(f"architecture field {key} must be a positive integer")
    return value


def parse_architecture_metadata(payload: object) -> ArchitectureMetadata:
    """Validate explicit architecture data; unknown model IDs are never substituted."""

    if not isinstance(payload, dict):
        raise ResearchEvidenceError("architecture metadata must be an object")
    strings: dict[str, str] = {}
    for key in ("model_id", "revision", "architecture", "source"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ResearchEvidenceError(f"architecture field {key} must be non-empty")
        strings[key] = value.strip()
    return ArchitectureMetadata(
        model_id=strings["model_id"],
        revision=strings["revision"],
        architecture=strings["architecture"],
        hidden_size=_positive_int(payload, "hidden_size"),
        num_hidden_layers=_positive_int(payload, "num_hidden_layers"),
        num_attention_heads=_positive_int(payload, "num_attention_heads"),
        num_key_value_heads=_positive_int(payload, "num_key_value_heads"),
        activation_dtype_bytes=_positive_int(payload, "activation_dtype_bytes"),
        source=strings["source"],
    )


def load_architecture_metadata(path: str | Path) -> ArchitectureMetadata:
    """Load one explicit metadata object."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchEvidenceError(f"cannot parse architecture metadata: {source}") from error
    return parse_architecture_metadata(payload)


def derive_hidden_state_payload_bytes(
    metadata: ArchitectureMetadata, *, instantaneous_batch_tokens: int
) -> int:
    """Derive one hidden-state collective payload from an observed token batch."""

    if (
        isinstance(instantaneous_batch_tokens, bool)
        or not isinstance(instantaneous_batch_tokens, int)
        or instantaneous_batch_tokens < 1
    ):
        raise ResearchEvidenceError("instantaneous_batch_tokens must be positive")
    return (
        instantaneous_batch_tokens
        * metadata.hidden_size
        * metadata.activation_dtype_bytes
    )
