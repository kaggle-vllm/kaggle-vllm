"""Inspection and metadata helpers for vLLM persistent sharded_state models."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .exceptions import ShardedModelError

SHARD_PATTERN = re.compile(r"^model-rank-(\d+)-part-(\d+)\.safetensors$")
REQUIRED_METADATA = ("config.json", "tokenizer_config.json")
WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}


@dataclass(frozen=True)
class ShardFile:
    """One rank-specific checkpoint part."""

    name: str
    rank: int
    part: int
    size: int


@dataclass(frozen=True)
class ShardedModelInspection:
    """Serializable structural inspection of a sharded_state directory."""

    path: str
    rank_count: int
    total_size: int
    shards: tuple[ShardFile, ...]
    metadata_files: tuple[str, ...]
    missing_metadata: tuple[str, ...]
    topology_errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            bool(self.shards) and not self.missing_metadata and not self.topology_errors
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid"] = self.valid
        return result


def _index_warnings(root: Path) -> list[str]:
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        return []
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        referenced = sorted(set(payload.get("weight_map", {}).values()))
    except (OSError, ValueError, AttributeError):
        return ["model.safetensors.index.json could not be parsed"]
    absent = [name for name in referenced if not (root / name).is_file()]
    if absent:
        return [
            (
                "The copied Hugging Face weight index references files absent from "
                "this directory; vLLM sharded_state loading uses rank-specific files "
                "instead."
            )
        ]
    return []


def inspect_sharded_model(
    path: str | Path,
    *,
    expected_tensor_parallel_size: int | None = None,
) -> ShardedModelInspection:
    """Inspect rank/part topology and small metadata without reading weight bodies."""

    expanded = Path(path).expanduser()
    lexical = Path(os.path.abspath(expanded))
    root = expanded.resolve(strict=False)
    if lexical != root:
        raise ShardedModelError(
            f"refusing sharded model path that traverses a symlink: {lexical} -> {root}"
        )
    if not root.is_dir():
        raise ShardedModelError(f"sharded model directory does not exist: {root}")
    if expected_tensor_parallel_size is not None and expected_tensor_parallel_size < 1:
        raise ShardedModelError("expected tensor parallel size must be positive")

    shards: list[ShardFile] = []
    metadata: list[str] = []
    for entry in sorted(root.iterdir()):
        if entry.is_symlink():
            raise ShardedModelError(
                f"refusing symlink in sharded model directory: {entry.name}"
            )
        if not entry.is_file():
            continue
        match = SHARD_PATTERN.match(entry.name)
        if match:
            shards.append(
                ShardFile(
                    name=entry.name,
                    rank=int(match.group(1)),
                    part=int(match.group(2)),
                    size=entry.stat().st_size,
                )
            )
        elif entry.suffix != ".safetensors":
            metadata.append(entry.name)

    if not shards:
        raise ShardedModelError(
            f"no rank-specific model-rank-<rank>-part-<part>.safetensors files in {root}"
        )

    warnings = _index_warnings(root)
    topology_errors: list[str] = []
    ranks = sorted({shard.rank for shard in shards})
    if ranks != list(range(len(ranks))):
        topology_errors.append(f"rank sequence is not contiguous from zero: {ranks}")
    if (
        expected_tensor_parallel_size is not None
        and len(ranks) != expected_tensor_parallel_size
    ):
        topology_errors.append(
            f"checkpoint has {len(ranks)} ranks but tensor parallel size "
            f"{expected_tensor_parallel_size} was requested"
        )
    reference_parts: list[int] | None = None
    for rank in ranks:
        parts = sorted(shard.part for shard in shards if shard.rank == rank)
        if parts != list(range(len(parts))):
            topology_errors.append(
                f"part sequence for rank {rank} is not contiguous: {parts}"
            )
        if reference_parts is None:
            reference_parts = parts
        elif parts != reference_parts:
            topology_errors.append(
                f"part sequence for rank {rank} differs from rank 0: {parts}"
            )

    missing = tuple(name for name in REQUIRED_METADATA if not (root / name).is_file())
    return ShardedModelInspection(
        path=str(root.resolve()),
        rank_count=len(ranks),
        total_size=sum(shard.size for shard in shards),
        shards=tuple(shards),
        metadata_files=tuple(metadata),
        missing_metadata=missing,
        topology_errors=tuple(topology_errors),
        warnings=tuple(warnings),
    )


def copy_model_metadata(
    source: str | Path, destination: str | Path
) -> tuple[Path, ...]:
    """Copy non-weight model/tokenizer metadata from a local HF snapshot."""

    src_expanded = Path(source).expanduser()
    dst_expanded = Path(destination).expanduser()
    src_root = src_expanded.resolve(strict=False)
    dst_root = dst_expanded.resolve(strict=False)
    if Path(os.path.abspath(src_expanded)) != src_root:
        raise ShardedModelError("refusing metadata source that traverses a symlink")
    if Path(os.path.abspath(dst_expanded)) != dst_root:
        raise ShardedModelError(
            "refusing metadata destination that traverses a symlink"
        )
    if not src_root.is_dir():
        return ()
    if not dst_root.is_dir():
        raise ShardedModelError(f"metadata destination is not a directory: {dst_root}")
    copied: list[Path] = []
    for source_path in src_root.iterdir():
        if source_path.is_symlink():
            raise ShardedModelError(
                f"refusing symlink in model metadata source: {source_path.name}"
            )
        if source_path.suffix.casefold() in WEIGHT_SUFFIXES:
            continue
        destination_path = dst_root / source_path.name
        if destination_path.is_symlink():
            raise ShardedModelError(
                f"refusing symlink in model metadata destination: {source_path.name}"
            )
        if source_path.is_dir():
            if any(path.is_symlink() for path in source_path.rglob("*")):
                raise ShardedModelError(
                    "refusing nested symlink in model metadata source: "
                    f"{source_path.name}"
                )
            if destination_path.exists():
                continue
            shutil.copytree(source_path, destination_path)
        else:
            shutil.copy2(source_path, destination_path)
        copied.append(destination_path)
    return tuple(copied)
