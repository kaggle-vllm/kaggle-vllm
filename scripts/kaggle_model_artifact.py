#!/usr/bin/env python3
"""Auditable one-model TP=2 sharded-state workflow for Kaggle T4x2."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle_vllm import inspect_sharded_model
from kaggle_vllm.research.errors import ResearchEvidenceError
from kaggle_vllm.research.provenance import sha256_file
from kaggle_vllm.research.resources import (
    GPU_MEMORY_LIMIT_MIB,
    ResourceMonitor,
    capture_resource_samples,
    directory_size,
    require_disk_budget,
)

EXPECTED_NATIVE_SHA256 = (
    "5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c"
)
EXPECTED_VLLM_DISTRIBUTION = "0.18.2.dev0+ga26e8dc7f.d20260822.cu128"
UPLOAD_REPOSITORY = "waqasm86/kaggle-vllm-models"
ALLOWED_UPLOAD_STATUS = "CONDITIONALLY_PERMITTED_PRESERVE_LICENSE_NOTICE"
WEIGHT_SUFFIXES = {".bin", ".pt", ".pth", ".safetensors"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_owned_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved == resolved_root or resolved_root not in resolved.parents:
        raise ResearchEvidenceError(f"path is outside the notebook-owned root: {path}")
    if path.is_symlink():
        raise ResearchEvidenceError(f"refusing symlinked workflow path: {path}")
    return resolved


def remove_owned(path: Path, root: Path) -> None:
    target = safe_owned_path(path, root)
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def load_model_config(repository: Path, model_key: str) -> dict[str, Any]:
    matrix = json.loads((repository / "research/model_matrix.json").read_text())
    model = matrix.get("models", {}).get(model_key)
    if not isinstance(model, dict):
        raise ResearchEvidenceError(f"unknown model key: {model_key}")
    if model_key == "qwen25_3b":
        raise ResearchEvidenceError("Qwen uses the preserved validated artifact")
    required = (
        "hf_id",
        "revision",
        "architecture",
        "selected_weight_bytes",
        "license",
        "hf_license_id",
        "license_url",
        "redistribution_status",
        "allow_patterns",
    )
    missing = [name for name in required if name not in model]
    if missing:
        raise ResearchEvidenceError(f"model matrix entry missing: {missing}")
    return model


def validate_runtime(native_wheel: Path) -> dict[str, Any]:
    import torch

    if importlib.metadata.version("kaggle-vllm") != "0.2.0":
        raise ResearchEvidenceError("the workflow requires kaggle-vllm 0.2.0")
    vllm_distribution = importlib.metadata.version("vllm")
    if vllm_distribution != EXPECTED_VLLM_DISTRIBUTION:
        raise ResearchEvidenceError(
            f"validated vLLM distribution required: {EXPECTED_VLLM_DISTRIBUTION}; "
            f"found {vllm_distribution}"
        )
    if torch.__version__ != "2.10.0+cu128" or torch.version.cuda != "12.8":
        raise ResearchEvidenceError("validated system Torch 2.10.0+cu128/CUDA 12.8 required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise ResearchEvidenceError("exactly two CUDA-visible GPUs are required")
    gpus = []
    for index in range(2):
        name = torch.cuda.get_device_name(index)
        capability = list(torch.cuda.get_device_capability(index))
        total = torch.cuda.get_device_properties(index).total_memory
        if name not in {"Tesla T4", "NVIDIA Tesla T4"} or capability != [7, 5]:
            raise ResearchEvidenceError(f"GPU {index} is not Tesla T4 SM75")
        gpus.append(
            {
                "index": index,
                "name": name,
                "compute_capability": capability,
                "total_memory_bytes": total,
                "maximum_allowed_memory_mib": GPU_MEMORY_LIMIT_MIB,
                "conservative_gpu_memory_utilization": min(
                    0.80, GPU_MEMORY_LIMIT_MIB * 1024**2 / total
                ),
            }
        )
    nccl = ".".join(str(value) for value in torch.cuda.nccl.version())
    if nccl != "2.27.5":
        raise ResearchEvidenceError(f"validated NCCL 2.27.5 required, found {nccl}")
    if not native_wheel.is_file() or sha256_file(native_wheel) != EXPECTED_NATIVE_SHA256:
        raise ResearchEvidenceError("validated native wheel SHA256 mismatch")
    identity = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if identity.returncode != 0 or len(identity.stdout.strip().splitlines()) != 2:
        raise ResearchEvidenceError("could not capture two-GPU nvidia-smi identity")
    if any("580.159.04" not in line for line in identity.stdout.splitlines()):
        raise ResearchEvidenceError("validated NVIDIA driver 580.159.04 required")
    toolkit = subprocess.run(
        ["nvcc", "--version"], capture_output=True, text=True, check=False
    )
    if toolkit.returncode != 0 or "V12.8.93" not in toolkit.stdout:
        raise ResearchEvidenceError("validated CUDA toolkit 12.8.93 required")
    topology = {}
    for name, command in {
        "matrix": ["nvidia-smi", "topo", "-m"],
        "p2p_read": ["nvidia-smi", "topo", "-p2p", "r"],
        "p2p_write": ["nvidia-smi", "topo", "-p2p", "w"],
    }.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ResearchEvidenceError(f"topology capture failed: {name}")
        topology[name] = result.stdout.strip()
    if "PHB" not in topology["matrix"]:
        raise ResearchEvidenceError("artifact workflow requires observed PHB topology")
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": nccl,
        "vllm_distribution": vllm_distribution,
        "vllm_source_commit": "a26e8dc7ff2111a005144d775ecf9cebf56c45b2",
        "native_wheel_sha256": EXPECTED_NATIVE_SHA256,
        "nvidia_smi_identity": identity.stdout.strip(),
        "cuda_toolkit": toolkit.stdout.strip(),
        "topology": topology,
        "gpus": gpus,
    }


def verify_hub_metadata(model: dict[str, Any], token: str | None) -> dict[str, Any]:
    from huggingface_hub import HfApi

    info = HfApi(token=token).model_info(model["hf_id"], revision=model["revision"])
    if info.sha != model["revision"]:
        raise ResearchEvidenceError("Hugging Face revision did not resolve exactly")
    card = info.card_data.to_dict() if info.card_data is not None else {}
    observed_license = card.get("license")
    if observed_license != model["hf_license_id"]:
        raise ResearchEvidenceError(
            "model-card license changed: "
            f"expected {model['hf_license_id']!r}, observed {observed_license!r}"
        )
    return {
        "hf_id": model["hf_id"],
        "requested_revision": model["revision"],
        "resolved_revision": info.sha,
        "gated": info.gated,
        "private": info.private,
        "license_declared_by_model_card": observed_license,
        "license_expected": model["license"],
        "license_url": model["license_url"],
        "redistribution_status": model["redistribution_status"],
        "access_token_present": bool(token),
        "access_is_not_redistribution_permission": True,
    }


def source_manifest(source: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ResearchEvidenceError(f"source snapshot contains symlink: {path}")
        if path.is_file() and ".cache" not in path.relative_to(source).parts:
            rows.append(
                {
                    "path": path.relative_to(source).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def copy_metadata(source: Path, destination: Path) -> None:
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        if ".cache" in relative.parts:
            continue
        if item.is_symlink():
            raise ResearchEvidenceError(f"metadata source is a symlink: {relative}")
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.suffix.casefold() not in WEIGHT_SUFFIXES:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def stage_save(args: argparse.Namespace) -> int:
    from vllm import LLM

    llm = LLM(
        model=str(args.source),
        tensor_parallel_size=2,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
    )
    llm.llm_engine.engine_core.save_sharded_state(
        path=str(args.shard_dir), max_size=2 * 1024**3
    )
    copy_metadata(args.source, args.shard_dir)
    print(json.dumps({"status": "SHARDED_STATE_SAVED", "path": str(args.shard_dir)}))
    return 0


def stage_download(args: argparse.Namespace) -> int:
    from huggingface_hub import snapshot_download

    repository = Path(__file__).resolve().parents[1]
    model = load_model_config(repository, args.model_key)
    resolved = Path(
        snapshot_download(
            repo_id=model["hf_id"],
            revision=model["revision"],
            local_dir=args.source,
            allow_patterns=model["allow_patterns"],
            token=os.environ.get("HF_TOKEN") or None,
            cache_dir=args.cache_dir,
        )
    ).resolve()
    if resolved != args.source.resolve():
        raise ResearchEvidenceError("snapshot escaped the notebook-owned source directory")
    print(json.dumps({"status": "PINNED_SOURCE_DOWNLOADED", "path": str(resolved)}))
    return 0


def stage_reload(args: argparse.Namespace) -> int:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.shard_dir),
        load_format="sharded_state",
        tensor_parallel_size=2,
        dtype="float16",
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        enable_prefix_caching=False,
    )
    outputs = llm.generate(
        ["State one reason measured evidence needs provenance."],
        SamplingParams(temperature=0.0, max_tokens=32),
    )
    result = {
        "status": "FRESH_SHARDED_STATE_RELOAD_PASS",
        "model_path": str(args.shard_dir),
        "prompt": outputs[0].prompt,
        "output": outputs[0].outputs[0].text,
        "output_token_ids": list(outputs[0].outputs[0].token_ids),
    }
    write_json(args.stage_output, result)
    return 0


def read_phase_file(path: Path) -> tuple[str, bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload["phase"]), bool(payload["active"])
    except (OSError, KeyError, json.JSONDecodeError):
        return "transition", False


def set_phase(path: Path, phase: str, active: bool) -> None:
    path.write_text(json.dumps({"phase": phase, "active": active}), encoding="utf-8")


def run_monitored(
    command: list[str],
    *,
    log_path: Path,
    phase_file: Path,
    phase: str,
    environment: dict[str, str],
) -> tuple[int, list[dict[str, Any]]]:
    set_phase(phase_file, phase, True)
    monitor = ResourceMonitor(phase_provider=lambda: read_phase_file(phase_file))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        )
        monitor.start()
        while process.poll() is None:
            time.sleep(0.25)
            if monitor.violation:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
        returncode = process.wait()
        monitor.stop()
    set_phase(phase_file, f"{phase}_complete", False)
    monitor.raise_if_violated()
    return returncode, [sample.to_dict() for sample in monitor.samples]


def query_json(url: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ResearchEvidenceError(f"API returned non-object JSON: {url}")
    return payload


def validate_api(
    shard_dir: Path,
    slug: str,
    output_dir: Path,
    phase_file: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    command = [
        "vllm",
        "serve",
        str(shard_dir),
        "--served-model-name",
        slug,
        "--load-format",
        "sharded_state",
        "--tensor-parallel-size",
        "2",
        "--dtype",
        "float16",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.80",
        "--enforce-eager",
        "--disable-custom-all-reduce",
        "--no-enable-prefix-caching",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    set_phase(phase_file, "api_validation", True)
    monitor = ResourceMonitor(phase_provider=lambda: read_phase_file(phase_file))
    log_path = output_dir / f"{slug}-t4x2-api-server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        )
        monitor.start()
        ready = False
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline and process.poll() is None:
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2).read()
                ready = True
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
            if monitor.violation:
                break
        if not ready:
            process.terminate()
            process.wait(timeout=30)
            monitor.stop()
            monitor.raise_if_violated()
            raise ResearchEvidenceError("OpenAI-compatible server did not become ready")
        models = query_json("http://127.0.0.1:8000/v1/models")
        completion = query_json(
            "http://127.0.0.1:8000/v1/completions",
            {
                "model": slug,
                "prompt": "State one reproducibility requirement:",
                "max_tokens": 16,
                "temperature": 0,
            },
        )
        write_json(output_dir / f"{slug}-t4x2-openai-models.json", models)
        write_json(output_dir / f"{slug}-t4x2-openai-completion.json", completion)
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        monitor.stop()
    set_phase(phase_file, "api_validation_complete", False)
    monitor.raise_if_violated()
    return {
        "command": command,
        "models_response_file": f"{slug}-t4x2-openai-models.json",
        "completion_response_file": f"{slug}-t4x2-openai-completion.json",
        "server_log": log_path.name,
        "termination_returncode": process.returncode,
        "telemetry": [sample.to_dict() for sample in monitor.samples],
    }


def verify_rank_logs(*paths: Path) -> dict[str, Any]:
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths)
    rank0 = bool(re.search(r"(?:rank=0|rank 0)", text))
    rank1 = bool(re.search(r"(?:rank=1|rank 1)", text))
    triton = "TRITON_ATTN" in text
    if not rank0 or not rank1 or not triton:
        raise ResearchEvidenceError(
            "logs did not prove both TP ranks and the validated TRITON_ATTN backend"
        )
    return {"rank0_observed": rank0, "rank1_observed": rank1, "triton_attn": triton}


def create_archive(shard_dir: Path, archive: Path) -> dict[str, Any]:
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(shard_dir, arcname=shard_dir.name, recursive=True)
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getnames()
    if not members or not all(
        name == shard_dir.name or name.startswith(f"{shard_dir.name}/")
        for name in members
    ):
        raise ResearchEvidenceError("archive member verification failed")
    return {
        "file": archive.name,
        "size_bytes": archive.stat().st_size,
        "sha256": sha256_file(archive),
        "member_count": len(members),
        "verified_without_extraction": True,
    }


def upload_and_verify(
    model: dict[str, Any],
    files: list[Path],
    *,
    slug: str,
    token: str,
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    remote_prefix = f"sharded-state/{slug}"
    commits = []
    for path in files:
        result = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{remote_prefix}/{path.name}",
            repo_id=UPLOAD_REPOSITORY,
            repo_type="model",
            commit_message=f"Add {slug} TP2 sharded-state research artifact",
        )
        commits.append(str(result.oid))
    tree = api.list_repo_tree(
        UPLOAD_REPOSITORY,
        path_in_repo=remote_prefix,
        recursive=False,
        expand=True,
        repo_type="model",
    )
    remote_sizes = {item.path: item.size for item in tree if hasattr(item, "size")}
    for path in files:
        remote = f"{remote_prefix}/{path.name}"
        if remote_sizes.get(remote) != path.stat().st_size:
            raise ResearchEvidenceError(f"remote size verification failed: {remote}")
    return {
        "repository": UPLOAD_REPOSITORY,
        "remote_prefix": remote_prefix,
        "commit_revisions": commits,
        "remote_sizes": remote_sizes,
        "verification": "remote object existence and exact size; no second full download",
        "license_status": model["redistribution_status"],
    }


def finalize_hashes(output_dir: Path) -> None:
    names = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(output_dir / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def orchestrate(args: argparse.Namespace) -> int:
    started = utc_now()
    repository = Path(__file__).resolve().parents[1]
    model = load_model_config(repository, args.model_key)
    slug = args.model_key.replace("_", "-")
    working_root = args.working_root.resolve()
    output_dir = working_root / f"{slug}-evidence"
    source = working_root / f"{slug}-source-model"
    shard_dir = working_root / f"{slug}-t4x2-sharded"
    archive = output_dir / f"{slug}-t4x2-sharded.tar.gz"
    for path in (output_dir, source, shard_dir):
        if path.exists():
            raise ResearchEvidenceError(f"refusing existing workflow path: {path}")
    output_dir.mkdir(parents=True)
    shard_dir.mkdir()
    phase_file = output_dir / ".phase.json"
    native_wheel = args.native_wheel.resolve()
    environment = validate_runtime(native_wheel)
    token = os.environ.get("HF_TOKEN") or None
    hub_identity = verify_hub_metadata(model, token)
    pre_download = require_disk_budget(
        working_root,
        projected_additional_bytes=int(model["selected_weight_bytes"]) + 1024**3,
    )
    source.mkdir()
    hf_cache = working_root / ".hf-home"
    child_environment = dict(os.environ)
    child_environment["HF_HOME"] = str(hf_cache)
    child_environment["TOKENIZERS_PARALLELISM"] = "false"
    download_log = output_dir / f"{slug}-source-download.log"
    download_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "download",
        "--model-key",
        args.model_key,
        "--source",
        str(source),
        "--cache-dir",
        str(hf_cache),
    ]
    download_rc, download_telemetry = run_monitored(
        download_command,
        log_path=download_log,
        phase_file=phase_file,
        phase="source_download",
        environment=child_environment,
    )
    if download_rc != 0:
        raise ResearchEvidenceError("pinned source-model download failed")
    source_files = source_manifest(source)
    actual_weight_bytes = sum(
        row["size_bytes"]
        for row in source_files
        if Path(row["path"]).suffix.casefold() in WEIGHT_SUFFIXES
    )
    if actual_weight_bytes != int(model["selected_weight_bytes"]):
        raise ResearchEvidenceError(
            f"source weight bytes differ from reviewed model matrix: {actual_weight_bytes}"
        )
    require_disk_budget(
        working_root,
        projected_additional_bytes=int(model["expected_sharded_bytes"]) + 512 * 1024**2,
    )
    save_log = output_dir / f"{slug}-t4x2-save.log"
    save_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "save",
        "--source",
        str(source),
        "--shard-dir",
        str(shard_dir),
    ]
    save_rc, save_telemetry = run_monitored(
        save_command,
        log_path=save_log,
        phase_file=phase_file,
        phase="tp2_save",
        environment=child_environment,
    )
    if save_rc != 0:
        raise ResearchEvidenceError("TP2 load/save subprocess failed")
    inspection = inspect_sharded_model(shard_dir, expected_tensor_parallel_size=2)
    if not inspection.valid:
        raise ResearchEvidenceError("saved sharded-state topology is invalid")
    shard_manifest = [
        {
            "filename": shard.name,
            "rank": shard.rank,
            "part": shard.part,
            "size_bytes": shard.size,
            "sha256": sha256_file(shard_dir / shard.name),
        }
        for shard in inspection.shards
    ]
    reload_output = output_dir / f"{slug}-t4x2-fresh-reload.json"
    reload_log = output_dir / f"{slug}-t4x2-fresh-reload.log"
    reload_environment = dict(child_environment)
    reload_environment["HF_HUB_OFFLINE"] = "1"
    reload_environment["TRANSFORMERS_OFFLINE"] = "1"
    reload_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "reload",
        "--shard-dir",
        str(shard_dir),
        "--stage-output",
        str(reload_output),
    ]
    reload_rc, reload_telemetry = run_monitored(
        reload_command,
        log_path=reload_log,
        phase_file=phase_file,
        phase="fresh_reload",
        environment=reload_environment,
    )
    if reload_rc != 0:
        raise ResearchEvidenceError("fresh sharded-state reload subprocess failed")
    api = validate_api(
        shard_dir, slug, output_dir, phase_file, reload_environment
    )
    log_observations = verify_rank_logs(save_log, reload_log, output_dir / api["server_log"])
    manifest_path = output_dir / f"{slug}-t4x2-sharded-manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": "kaggle-vllm-tp2-sharded-manifest-v1",
            "artifact_type": "vllm-native-topology-specific-sharded-state",
            "not_a_transformers_checkpoint": True,
            "model": model,
            "tensor_parallel_size": 2,
            "part_size_limit_bytes": 2 * 1024**3,
            "inspection": inspection.to_dict(),
            "shards": shard_manifest,
            "source_files": source_files,
        },
    )
    notice_path = output_dir / f"{slug}-LICENSE-ATTRIBUTION.md"
    notice_path.write_text(
        "# Model-derived artifact attribution\n\n"
        f"- Source model: `{model['hf_id']}`\n"
        f"- Exact revision: `{model['revision']}`\n"
        f"- Declared license: {model['license']}\n"
        f"- License/terms: {model['license_url']}\n"
        "- Artifact type: vLLM-native TP=2 topology-specific sharded state; "
        "not a trained, fine-tuned, or ordinary Transformers checkpoint.\n\n"
        "The source model's license and attribution files are preserved inside "
        "the archive. Token access does not establish redistribution permission.\n",
        encoding="utf-8",
    )
    remove_owned(source, working_root)
    if hf_cache.exists():
        remove_owned(hf_cache, working_root)
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    projected_archive = max(directory_size(shard_dir), int(model["selected_weight_bytes"]))
    require_disk_budget(
        working_root, projected_additional_bytes=projected_archive + 256 * 1024**2
    )
    archive_identity = create_archive(shard_dir, archive)
    sha_path = output_dir / f"{slug}-t4x2-sharded.tar.gz.sha256"
    sha_path.write_text(f"{archive_identity['sha256']}  {archive.name}\n", encoding="utf-8")
    validation_path = output_dir / f"{slug}-t4x2-validation.json"
    validation = {
        "schema_version": "kaggle-vllm-tp2-sharded-validation-v1",
        "status": "KAGGLE_VALIDATED",
        "environment": environment,
        "hub_identity": hub_identity,
        "pre_download_disk_guard": pre_download,
        "sharded_inspection": inspection.to_dict(),
        "fresh_reload": json.loads(reload_output.read_text()),
        "api": api,
        "runtime_log_observations": log_observations,
        "resource_telemetry": {
            "download": download_telemetry,
            "save": save_telemetry,
            "reload": reload_telemetry,
            "api": api.pop("telemetry"),
        },
        "archive": archive_identity,
    }
    write_json(validation_path, validation)
    provenance_path = output_dir / f"{slug}-t4x2-provenance.json"
    notebook_sha = sha256_file(args.notebook_source.resolve())
    provenance = {
        "schema_version": "kaggle-vllm-tp2-sharded-provenance-v1",
        "started_at_utc": started,
        "completed_at_utc": utc_now(),
        "kaggle_vllm_version": "0.2.0",
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "repository_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repository, text=True
        ).strip(),
        "repository_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repository, text=True
            ).strip()
        ),
        "model_id": model["hf_id"],
        "model_revision": model["revision"],
        "tokenizer_revision": model["revision"],
        "license": model["license"],
        "redistribution_status": model["redistribution_status"],
        "native_wheel_sha256": EXPECTED_NATIVE_SHA256,
        "environment": environment,
        "notebook": str(args.notebook_source),
        "notebook_sha256": notebook_sha,
        "generation_command": sys.argv,
        "subprocess_commands": {
            "download": download_command,
            "save": save_command,
            "fresh_reload": reload_command,
            "api": api["command"],
        },
        "random_seed": 0,
        "credentials_recorded": False,
    }
    write_json(provenance_path, provenance)
    upload = None
    if model["redistribution_status"] == ALLOWED_UPLOAD_STATUS:
        if not token:
            raise ResearchEvidenceError("HF_TOKEN is required for permitted upload")
        upload_files = [archive, sha_path, manifest_path, validation_path, provenance_path]
        upload_files.append(notice_path)
        require_disk_budget(working_root, projected_additional_bytes=0)
        upload = upload_and_verify(model, upload_files, slug=slug, token=token)
        validation["upload"] = upload
        validation["status"] = "KAGGLE_VALIDATED_AND_UPLOAD_VERIFIED"
        remove_owned(archive, working_root)
        remove_owned(shard_dir, working_root)
        cleanup_status = "LARGE_LOCAL_ARTIFACTS_REMOVED_AFTER_REMOTE_VERIFICATION"
    else:
        validation["upload"] = {
            "status": "UPLOAD_BLOCKED_LICENSE_REVIEW",
            "reason": model["redistribution_status"],
        }
        write_json(validation_path, validation)
        cleanup_status = "LOCAL_ARTIFACT_RETAINED_BECAUSE_UPLOAD_NOT_VERIFIED"
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_samples = capture_resource_samples(
        phase="post_cleanup", measurement_active=False
    )
    validation["post_cleanup"] = {
        "status": cleanup_status,
        "working_directory_bytes": directory_size(working_root),
        "resources": [sample.to_dict() for sample in cleanup_samples],
    }
    write_json(validation_path, validation)
    provenance["upload"] = upload
    provenance["cleanup_status"] = cleanup_status
    provenance["completed_at_utc"] = utc_now()
    provenance["final_artifact_hashes_excluding_provenance"] = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
        and not path.name.startswith(".")
        and path.name not in {provenance_path.name, "SHA256SUMS.txt"}
    }
    write_json(provenance_path, provenance)
    if upload is not None:
        metadata_refresh = upload_and_verify(
            model,
            [validation_path, provenance_path],
            slug=slug,
            token=token,
        )
        provenance["metadata_refresh_commit_revisions"] = metadata_refresh[
            "commit_revisions"
        ]
        write_json(provenance_path, provenance)
        final_refresh = upload_and_verify(
            model, [provenance_path], slug=slug, token=token
        )
        # Do not rewrite provenance after this point: its remote size is now verified.
        upload["final_provenance_commit_revision"] = final_refresh[
            "commit_revisions"
        ][0]
    phase_file.unlink(missing_ok=True)
    finalize_hashes(output_dir)
    print(json.dumps({"status": validation["status"], "evidence": str(output_dir)}, indent=2))
    return 0


def record_failure(args: argparse.Namespace, error: Exception) -> None:
    if args.working_root is None or args.model_key is None:
        return
    output = args.working_root.resolve() / f"{args.model_key.replace('_', '-')}-evidence"
    output.mkdir(parents=True, exist_ok=True)
    message = str(error)
    token = os.environ.get("HF_TOKEN")
    if token:
        message = message.replace(token, "[REDACTED]")
    slug = args.model_key.replace("_", "-")
    validation_path = output / f"{slug}-t4x2-validation.json"
    write_json(
        validation_path,
        {
            "schema_version": "kaggle-vllm-tp2-sharded-validation-v1",
            "status": "UNSUPPORTED_OR_FAILED",
            "observed_error_type": type(error).__name__,
            "observed_error": message[:4000],
            "classification": "OBSERVED_NEGATIVE_RESULT",
            "fabricated_success": False,
            "captured_at_utc": utc_now(),
        },
    )
    repository = Path(__file__).resolve().parents[1]
    model = load_model_config(repository, args.model_key)
    provenance = {
        "schema_version": "kaggle-vllm-tp2-sharded-provenance-v1",
        "status": "KAGGLE_RUN_FAILED",
        "captured_at_utc": utc_now(),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip(),
        "repository_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repository, text=True
        ).strip(),
        "repository_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=repository, text=True
            ).strip()
        ),
        "kaggle_vllm_version": "0.2.0",
        "model_id": model["hf_id"],
        "model_revision": model["revision"],
        "tokenizer_revision": model["revision"],
        "license": model["license"],
        "redistribution_status": model["redistribution_status"],
        "native_wheel_sha256": (
            sha256_file(args.native_wheel)
            if args.native_wheel is not None and args.native_wheel.is_file()
            else None
        ),
        "notebook_sha256": (
            sha256_file(args.notebook_source)
            if args.notebook_source is not None and args.notebook_source.is_file()
            else None
        ),
        "generation_command": sys.argv,
        "validation_sha256": sha256_file(validation_path),
        "credentials_recorded": False,
    }
    write_json(output / f"{slug}-t4x2-provenance.json", provenance)
    finalize_hashes(output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key")
    parser.add_argument("--working-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--native-wheel", type=Path)
    parser.add_argument("--notebook-source", type=Path)
    parser.add_argument("--stage", choices=("download", "save", "reload"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--stage-output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "save" and (args.source is None or args.shard_dir is None):
        parser.error("save stage requires --source and --shard-dir")
    if args.stage == "download" and (
        args.model_key is None or args.source is None or args.cache_dir is None
    ):
        parser.error("download stage requires model key, source, and cache directory")
    if args.stage == "reload" and (
        args.shard_dir is None or args.stage_output is None
    ):
        parser.error("reload stage requires --shard-dir and --stage-output")
    if args.stage is None and (
        args.model_key is None or args.native_wheel is None or args.notebook_source is None
    ):
        parser.error("orchestration requires model key, native wheel, and notebook source")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "download":
        return stage_download(args)
    if args.stage == "save":
        return stage_save(args)
    if args.stage == "reload":
        return stage_reload(args)
    evidence = (
        args.working_root.resolve()
        / f"{args.model_key.replace('_', '-')}-evidence"
    )
    created_here = not evidence.exists()
    try:
        return orchestrate(args)
    except Exception as error:
        if created_here:
            record_failure(args, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
