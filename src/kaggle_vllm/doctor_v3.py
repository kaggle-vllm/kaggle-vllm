"""Targeted L4/cu129 runtime diagnostics; no global pip-check gate."""
from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .environment import collect
from .l4_bootstrap import DEFAULT_TARGET
from .profile_v3 import load_profile
from .runtime import all_nvidia_l4, all_sm89
from .runtime_v3 import TORCHCODEC_AUDIT_SHA256, sha256, verify_runtime
from .cuda_driver import DEFAULT_DRIVER

EXPECTED = {"vllm": "0.30.0", "torch": "2.13.0+cu129", "torchvision": "0.28.0+cu129", "torchaudio": "2.11.0+cu129", "torchcodec": "0.16.0+cu129", "triton": "3.7.1", "flashinfer-python": "0.6.18.post1", "flashinfer-cubin": "0.6.18.post1"}


def doctor_report(*, target: str | Path = DEFAULT_TARGET, strict: bool = True, import_native: bool = False) -> dict[str, Any]:
    """Collect serializable findings; native imports are explicit and cp312 only."""
    p = load_profile()
    bootstrap_path = Path(target).resolve() / "manifests" / "sdk-runtime.json"
    if bootstrap_path.is_file() and sys.version_info[:2] == (3, 12):
        from .l4_bootstrap import activate_runtime
        activate_runtime(target)
    env = collect()
    findings: list[dict[str, str]] = []
    def check(name: str, ok: bool, actual: object) -> None:
        findings.append({"check": name, "status": "pass" if ok else "error", "actual": str(actual)})
    from . import __version__
    check("SDK version", __version__ == "0.3.1", __version__)
    check("Kaggle", env.is_kaggle, env.is_kaggle)
    check("Python ABI", sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 12), sys.version.split()[0])
    check("platform", platform.system() == "Linux" and platform.machine() == "x86_64", platform.platform())
    check("torch", env.torch == p.torch_version, env.torch)
    check("torch CUDA", env.torch_cuda == p.torch_cuda, env.torch_cuda)
    check("CUDA usable", env.cuda_available, env.cuda_available)
    check("GPU count", env.gpu_count == 4, env.gpu_count)
    check("GPU model", all_nvidia_l4(env.gpus), env.gpu_names)
    check("SM89", all_sm89(env.gpus), env.gpu_caps)
    check("CUDA driver", env.cuda_driver is not None, env.cuda_driver)
    dest = Path(target).resolve()
    ancestor = dest
    while not ancestor.exists(): ancestor = ancestor.parent
    free_bytes = shutil.disk_usage(ancestor).free
    check("scratch runtime root", Path("/kaggle/working") not in dest.parents and dest != Path("/kaggle/working"), dest)
    findings.append({"check": "free storage", "status": "pass" if free_bytes >= 24 * 1024**3 or bootstrap_path.is_file() else "warning", "actual": str(free_bytes) + " bytes"})
    check("TP target", p.tensor_parallel_size == 4, p.tensor_parallel_size)
    check("NCCL version", getattr(env, "nccl", None) == "2.29.7", getattr(env, "nccl", None))
    pydeps_for_conflict = dest / "pydeps"
    preloaded = [name for name in ("torch", "vllm", "triton", "torchcodec", "flashinfer") if name in sys.modules and getattr(sys.modules[name], "__file__", None) and pydeps_for_conflict not in Path(sys.modules[name].__file__).resolve().parents]
    check("preloaded runtime conflicts", not preloaded, preloaded)
    bootstrap_manifest = dest / "manifests" / "sdk-runtime.json"
    check("bootstrap manifest", bootstrap_manifest.is_file(), bootstrap_manifest)
    if bootstrap_manifest.is_file():
        try:
            payload = json.loads(bootstrap_manifest.read_text(encoding="utf-8"))
            check("native wheel SHA256", payload.get("native_sha256") == p.wheel_sha256, payload.get("native_sha256"))
            source = Path(payload["source"])
            verify_runtime(source)
            check("runtime manifest integrity", True, source)
        except Exception as error:
            check("runtime manifest integrity", False, error)
    alias = dest / "cuda-driver-link" / "libcuda.so"
    check("libcuda linker alias", alias.is_symlink() and alias.resolve() == DEFAULT_DRIVER.resolve() and DEFAULT_DRIVER.is_file(), alias)
    if bootstrap_manifest.is_file():
        try:
            payload = json.loads(bootstrap_manifest.read_text(encoding="utf-8"))
            probe = payload.get("driver", {}).get("link_probe")
            if probe == "g++ unavailable":
                findings.append({"check": "libcuda link probe", "status": "warning", "actual": probe})
            else:
                check("libcuda link probe", probe == "PASS", probe)
        except (OSError, ValueError):
            pass
    pydeps = dest / "pydeps"
    versions = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in metadata.distributions(path=[str(pydeps)])} if pydeps.is_dir() else {}
    for name, expected in EXPECTED.items():
        check("distribution " + name, versions.get(name) == expected, versions.get(name))
    if (import_native or strict) and sys.version_info[:2] == (3, 12) and bootstrap_manifest.is_file():
        from .l4_bootstrap import activate_runtime
        activate_runtime(dest)
        for name in ("vllm._C_stable_libtorch", "vllm.vllm_flash_attn._vllm_fa2_C"):
            try:
                importlib.import_module(name)
                check("native import " + name, True, "PASS")
            except (ImportError, OSError) as error:
                check("native import " + name, False, error)
    elif strict:
        check("native core and FA2 imports", False, "requires staged cp312 runtime")
    if bootstrap_manifest.is_file():
        try:
            source = Path(json.loads(bootstrap_manifest.read_text(encoding="utf-8"))["source"])
            audit_path = source / "metadata" / "TORCHCODEC_ELF_AUDIT.json"
            if sha256(audit_path) != TORCHCODEC_AUDIT_SHA256:
                raise ValueError("TorchCodec ELF audit SHA256 mismatch")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            check("CUDA-13 ELF contamination", audit.get("cuda13_runtime_dependency") is False, audit)
        except (OSError, ValueError, KeyError) as error:
            check("CUDA-13 ELF contamination", False, error)
    return {"schema_version": 3, "profile": p.name, "strict": strict, "compatible": not any(x["status"] == "error" for x in findings), "findings": findings, "warnings": ["Global pip check is excluded from the profile gate; P31 recorded unrelated Kaggle package conflicts."], "evidence": {"nccl_tp4_p31": "PASS", "gemma4_semantic_step20_p31": "PASS", "fresh_session_acceptance": "historical_candidate_wheel_only", "FA2_NATIVE_IMPORT_PASS": "historical_candidate", "TRITON_ATTN_GEMMA_RUNTIME_PASS": "historical_candidate", "FA2_MODEL_RUNTIME_PASS": False}}


def run_doctor(*, target: str | Path = DEFAULT_TARGET, strict: bool = True, as_json_output: bool = False, import_native: bool = False) -> int:
    report = doctor_report(target=target, strict=strict, import_native=import_native)
    if as_json_output:
        print(json.dumps(report, indent=2))
    else:
        print(f"kaggle-vllm doctor: {report['profile']}")
        for item in report["findings"]:
            print(f"{item['status'].upper():5} {item['check']}: {item['actual']}")
    return 0 if report["compatible"] else 1
