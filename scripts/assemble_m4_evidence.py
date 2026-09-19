#!/usr/bin/env python3
"""Verify and assemble independently downloaded M4 Kaggle shard directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from kaggle_vllm.research.crossover import (
    M4_RAW_SCHEMA,
    PERFORMANCE_FIELDS,
    analyze_m4,
)
from kaggle_vllm.research.errors import ResearchEvidenceError
from kaggle_vllm.research.resources import GPU_MEMORY_LIMIT_MIB

CHECKSUM = re.compile(r"^(?P<digest>[0-9a-f]{64})  (?P<name>[^/\\]+)$")
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_shard(path: Path) -> dict[str, Any]:
    """Verify one flat shard checksum manifest and return its raw payload."""

    manifest = path / "SHA256SUMS.txt"
    if not path.is_dir() or not manifest.is_file():
        raise ResearchEvidenceError(f"missing M4 shard or checksum manifest: {path}")
    checked = 0
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = CHECKSUM.fullmatch(line)
        if not match:
            raise ResearchEvidenceError(f"malformed checksum line {line_number}: {path}")
        candidate = path / match.group("name")
        if not candidate.is_file() or candidate.is_symlink():
            raise ResearchEvidenceError(f"missing or unsafe checksummed file: {candidate}")
        if _sha256(candidate) != match.group("digest"):
            raise ResearchEvidenceError(f"checksum mismatch: {candidate}")
        checked += 1
    if checked == 0:
        raise ResearchEvidenceError(f"empty checksum manifest: {manifest}")
    raw_path = path / "m4-raw.json"
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchEvidenceError(f"invalid M4 raw file: {raw_path}") from error
    if payload.get("schema_version") != M4_RAW_SCHEMA or not isinstance(payload.get("rows"), list):
        raise ResearchEvidenceError(f"unsupported M4 raw payload: {raw_path}")
    summary_path = path / "execution-summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else None
    )
    provenance_path = path / "batch-shard-provenance.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.is_file()
        else None
    )
    return {
        "directory": str(path.resolve()),
        "manifest_sha256": _sha256(manifest),
        "raw_sha256": _sha256(raw_path),
        "execution_status": payload.get("status"),
        "session_provenance": provenance,
        "summary": summary,
        "payload": payload,
    }


def _validate_resource_boundary_shard(
    payload: dict[str, Any], summary: dict[str, Any] | None
) -> None:
    """Require the preserved, non-OOM terminal VRAM shape before analysis."""

    rows = payload.get("rows", [])
    failed = [row for row in rows if row.get("request_failures") or row.get("oom")]
    if (
        payload.get("status") != "RESOURCE_GUARD_VIOLATION"
        or payload.get("mode") != "principal"
        or not isinstance(summary, dict)
        or summary.get("status") != "RESOURCE_GUARD_VIOLATION"
        or summary.get("expected_cells") != 12
        or summary.get("completed_or_preserved_cells") != 12
        or summary.get("failed_cells") != len(failed)
        or not summary.get("resource_guard_violation")
        or summary.get("semantic_gate_failure") is not None
        or not failed
        or any(
            row.get("oom") is not False
            or row.get("maximum_vram_mib", 0) <= GPU_MEMORY_LIMIT_MIB
            or any(row.get(field) is not None for field in PERFORMANCE_FIELDS)
            for row in failed
        )
    ):
        raise ResearchEvidenceError("invalid terminal M4 resource-boundary shard")


def _source_freeze(
    repository: Path, source_commit: str
) -> tuple[str, dict[str, Any]]:
    candidates = [
        repository / "research/M4_SOURCE_FREEZE.json",
        *sorted(repository.glob("research/M4_BATCH_SOURCE_FREEZE*.json")),
    ]
    matches = []
    for path in candidates:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("implementation_source_commit") == source_commit:
            matches.append((path, value))
    if len(matches) != 1:
        raise ResearchEvidenceError(
            f"source commit lacks one immutable M4 freeze: {source_commit}"
        )
    path, freeze = matches[0]
    identity = {
        "execution_plan_sha256": freeze.get(
            "m4_execution_plan_sha256", freeze.get("execution_plan_sha256")
        ),
        "model_matrix_sha256": freeze.get("model_matrix_sha256"),
        "protocol_sha256": freeze.get("protocol_sha256"),
    }
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in identity.values()
    ):
        raise ResearchEvidenceError(f"incomplete M4 scientific freeze: {path}")
    return str(path.relative_to(repository)), identity


def assemble(
    shards: list[Path], *, repository: Path = ROOT
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Combine verified rows and run the fail-closed repeated-cell analyzer."""

    verified = [verify_shard(path.resolve()) for path in shards]
    rows: list[dict[str, Any]] = []
    sources = set()
    source_freezes: dict[str, str] = {}
    scientific_identities: set[tuple[str, str, str]] = set()
    artifacts = []
    resource_boundary_shards = 0
    for shard in verified:
        payload = shard.pop("payload")
        status = payload.get("status")
        if status == "RESOURCE_GUARD_VIOLATION":
            _validate_resource_boundary_shard(payload, shard.get("summary"))
            resource_boundary_shards += 1
        elif status != "executed" or payload.get("mode") not in {
            "principal", "refinement"
        }:
            raise ResearchEvidenceError(
                f"non-principal/refinement or incomplete shard: {shard['directory']}"
            )
        source = payload.get("source", {})
        if source.get("dirty") is not False or not source.get("commit"):
            raise ResearchEvidenceError(f"unclean or absent source identity: {shard['directory']}")
        commit = source["commit"]
        freeze_path, scientific_identity = _source_freeze(repository, commit)
        sources.add(commit)
        source_freezes[commit] = freeze_path
        scientific_identities.add(tuple(scientific_identity.values()))
        rows.extend(payload["rows"])
        artifacts.append(shard)
    if len(scientific_identities) != 1:
        raise ResearchEvidenceError(
            "M4 shards do not share one frozen scientific plan/model/protocol identity"
        )
    scientific_identity = scientific_identities.pop()
    combined = {
        "schema_version": M4_RAW_SCHEMA,
        "status": "ASSEMBLED_VERIFIED_SHARDS",
        "source_commits": sorted(sources),
        "source_freezes": source_freezes,
        "scientific_identity": {
            "execution_plan_sha256": scientific_identity[0],
            "model_matrix_sha256": scientific_identity[1],
            "protocol_sha256": scientific_identity[2],
        },
        "canonical_shards": len(artifacts) - resource_boundary_shards,
        "resource_boundary_shards": resource_boundary_shards,
        "shards": artifacts,
        "rows": rows,
    }
    if len(sources) == 1:
        combined["source_commit"] = next(iter(sources))
    # analyze_m4 deliberately reparses a file; validate through a temporary-shaped
    # serialized payload at the CLI boundary below rather than duplicating rules.
    return combined, {"pending_file_validation": True}


def _write_new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite output directory: {output}")
    combined, _ = assemble(args.shards)
    raw_path = output / "M4_RAW_COMBINED.json"
    _write_new(raw_path, combined)
    try:
        analysis = analyze_m4(raw_path)
    except Exception:
        # Do not leave a misleading partially assembled package.
        raw_path.unlink()
        output.rmdir()
        raise
    _write_new(output / "M4_ANALYSIS.json", analysis)
    files = sorted(output.iterdir(), key=lambda item: item.name)
    with (output / "SHA256SUMS.txt").open("x", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha256(path)}  {path.name}\n")
    print(json.dumps({"status": "ASSEMBLED_AND_ANALYZED", "output": str(output), "rows": len(combined["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
