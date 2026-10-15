"""Build and verify the exact P31 runtime-v3 tar with Python 3.11 only.

No wheel is imported, installed, changed, or downloaded by this script.
"""
from __future__ import annotations

import collections
import email
import hashlib
import io
import json
import os
import pathlib
import tarfile
import tempfile
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = pathlib.Path(__file__).resolve().parents[1]
BASE = REPO.parent
HOUSE = BASE / "kaggle-vllm-v030-wheelhouse-cp312-cu129-v2"
P31 = BASE / "kaggle-vllm-030-final-cuda-binaries-with-v2" / "p31-kaggle-vllm-v030-l4x4-final-v2.tar.gz"
NAME = "kaggle-vllm-v030-l4x4-runtime-v3"
WHEEL = "vllm-0.30.0-cp312-cp312-linux_x86_64.whl"
WHEEL_SHA = "9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8"
P31_SHA = "e82085d0c6f180d2f4f8eb8f52332a92051273715e0923082fc65fc8857ddad5"
ROOTS = ["vllm", "torch", "torchvision", "torchaudio", "torchcodec", "triton", "flashinfer-python", "flashinfer-cubin"]
EVIDENCE = ["06_torch_l4_validation.json", "10_build_contract.json", "11_build_result.json", "12_wheel_native_audit.json", "14_native_import_fa2_validation.json", "15_nccl_tp4_validation.json", "16B_cuda_driver_linker_fix.json", "19_final_cuda_binary_qualification.json", "20_gemma4_tp4_semantic_chat.json", "21_release_forensic_audit.json"]


def digest_stream(f):
    h = hashlib.sha256()
    while b := f.read(1024 * 1024):
        h.update(b)
    return h.hexdigest()


def digest(path):
    with path.open("rb") as f: return digest_stream(f)


def add_bytes(tar, relative, content):
    data = content if isinstance(content, bytes) else content.encode("utf-8")
    info = tarfile.TarInfo(NAME + "/" + relative)
    info.size = len(data)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(data).hexdigest(), len(data)


def main():
    if digest(P31) != P31_SHA or P31.stat().st_size != 291899459:
        raise RuntimeError("P31 final-v2 identity mismatch")
    house_manifest = json.loads((HOUSE / "WHEELHOUSE_MANIFEST.json").read_text())
    inventory = {}
    for item in house_manifest["wheels"]:
        path = HOUSE / item["package_group"] / item["filename"]
        if not path.is_file() or path.stat().st_size != item["size"] or digest(path) != item["SHA256"]:
            raise RuntimeError(f"wheelhouse hash/size mismatch: {path}")
        with zipfile.ZipFile(path) as z:
            meta = email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))))
        key = canonicalize_name(meta["Name"])
        if key in inventory: raise RuntimeError(f"duplicate distribution: {key}")
        inventory[key] = {"distribution": key, "version": meta["Version"], "filename": path.name, "size": path.stat().st_size, "sha256": item["SHA256"], "group": item["package_group"], "path": path, "requires": meta.get_all("Requires-Dist", [])}
    with tarfile.open(P31, "r:gz") as p31:
        native_member = next(m for m in p31 if m.name.endswith(".whl"))
        if pathlib.Path(native_member.name).name != WHEEL or native_member.size != 293855397:
            raise RuntimeError("P31 native member identity mismatch")
        native_bytes = p31.extractfile(native_member).read()
        if hashlib.sha256(native_bytes).hexdigest() != WHEEL_SHA:
            raise RuntimeError("P31 native SHA mismatch")
        with zipfile.ZipFile(io.BytesIO(native_bytes)) as z:
            if z.testzip() is not None: raise RuntimeError("invalid P31 wheel ZIP")
            members = z.namelist()
            if "vllm/_C_stable_libtorch.abi3.so" not in members or "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so" not in members:
                raise RuntimeError("P31 native members missing")
            meta = email.message_from_bytes(z.read(next(n for n in members if n.endswith(".dist-info/METADATA"))))
            inventory["vllm"] = {"distribution": "vllm", "version": meta["Version"], "filename": WHEEL, "size": len(native_bytes), "sha256": WHEEL_SHA, "group": "native", "path": None, "requires": meta.get_all("Requires-Dist", [])}
        evidence = {}
        for m in p31:
            if pathlib.Path(m.name).name in EVIDENCE:
                evidence[pathlib.Path(m.name).name] = p31.extractfile(m).read()
        # tar iteration above starts after the wheel. Reopen for evidence.
    with tarfile.open(P31, "r:gz") as p31:
        for m in p31:
            if pathlib.Path(m.name).name in EVIDENCE:
                evidence[pathlib.Path(m.name).name] = p31.extractfile(m).read()
    if set(evidence) != set(EVIDENCE): raise RuntimeError("required P31 evidence missing")
    queue = collections.deque((canonicalize_name(x), "") for x in ROOTS)
    seen = set(); closure = set(); edges = []; parents = {}
    while queue:
        name, extra = queue.popleft()
        if (name, extra) in seen: continue
        seen.add((name, extra)); closure.add(name)
        if name not in inventory: raise RuntimeError(f"missing runtime distribution: {name}")
        for raw in inventory[name]["requires"]:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({"python_version": "3.12", "python_full_version": "3.12.13", "sys_platform": "linux", "platform_machine": "x86_64", "extra": extra}): continue
            child = canonicalize_name(req.name)
            if child not in inventory: raise RuntimeError(f"unavailable requirement {name}: {raw}")
            if inventory[child]["version"] not in req.specifier:
                raise RuntimeError(f"version mismatch {name}: {raw}, found {inventory[child]['version']}")
            edges.append({"from": name, "from_extra": extra, "to": child, "requirement": raw})
            for next_extra in ("", *sorted(req.extras)):
                parents.setdefault((child, next_extra), {"from": name, "requirement": raw})
                queue.append((child, next_extra))
    nv = "nvidia-cuda-nvdisasm"
    if nv not in closure: raise RuntimeError("nvdisasm unexpectedly unreachable; audit disposition changed")
    if "torchcodec" not in closure or inventory["torchcodec"]["version"] != "0.16.0+cu129":
        raise RuntimeError("corrected TorchCodec absent")
    if any("torchcodec-0.17" in inventory[x]["filename"].lower() for x in closure):
        raise RuntimeError("old TorchCodec included")
    if house_manifest["cuda13_runtime_audit"] != {"contamination_count": 0, "direct_elf_dependencies": 0}:
        raise RuntimeError("CUDA-13 ELF audit changed")
    records = [{k: v[k] for k in ("distribution", "version", "filename", "size", "sha256", "group")} for key, v in sorted(inventory.items()) if key in closure and key != "vllm"]
    excluded = [{"distribution": key, "filename": v["filename"], "group": v["group"], "reason": "not reachable from runtime roots under CPython 3.12/Linux x86_64 markers and requested extras"} for key, v in sorted(inventory.items()) if key not in closure]
    graph = {"schema_version": 3, "environment": "CPython 3.12.13 Linux x86_64", "roots": ROOTS, "reachable_count_including_native": len(closure), "edges": edges, "nvdisasm": {"reachable": True, "via": "nvidia-cutlass-dsl -> nvidia-cutlass-dsl-libs-base -> nvidia-cuda-nvdisasm>=13.3,<14", "version": inventory[nv]["version"], "cuda13_direct_elf_dependencies": 0, "warning": "Retained as a metadata dependency; fresh Kaggle isolated staging requires acceptance."}}
    runtime = {"schema_version": 3, "profile": "kaggle-l4x4-cu129", "native": {"filename": WHEEL, "size": len(native_bytes), "sha256": WHEEL_SHA}, "wheel_count": len(records), "wheels": records, "cuda13_runtime_contamination_count": 0, "fresh_kaggle_isolated_staging": "pending"}
    status = {"release": "kaggle-vllm 0.3.0 runtime-v3 candidate", "p31_final_v2_sha256": P31_SHA, "native_sha256": WHEEL_SHA, "p31_semantic_gemma_step20": "PASS", "isolated_bootstrap_fresh_kaggle": "PENDING", "optional_fa2_model_e2e": "PENDING"}
    metadata_files = {
        "WHEELHOUSE_MANIFEST.json": (HOUSE / "WHEELHOUSE_MANIFEST.json").read_bytes(),
        "RUNTIME_MANIFEST.json": json.dumps(runtime, indent=2).encode(),
        "KAGGLE_INSTALL_REQUIREMENTS.txt": (HOUSE / "KAGGLE_INSTALL_REQUIREMENTS.txt").read_bytes(),
        "SOURCE_IDENTITY.json": (HOUSE / "SOURCE_IDENTITY.json").read_bytes(),
        "TORCHCODEC_ELF_AUDIT.json": json.dumps(house_manifest["torchcodec_elf_audit"], indent=2).encode(),
        "RUNTIME_DEPENDENCY_GRAPH.json": json.dumps(graph, indent=2).encode(),
        "RUNTIME_EXCLUSIONS.json": json.dumps({"schema_version": 3, "excluded": excluded, "nvdisasm_disposition": "included because reachable"}, indent=2).encode(),
        "RELEASE_STATUS.json": json.dumps(status, indent=2).encode(),
    }
    target = REPO / "dist" / (NAME + ".tar.gz")
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    expected = {}
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=1) as tar:
            expected["native/" + WHEEL] = (WHEEL_SHA, len(native_bytes))
            add_bytes(tar, "native/" + WHEEL, native_bytes)
            for record in records:
                path = inventory[record["distribution"]]["path"]
                rel = "wheelhouse/" + record["filename"]
                tar.add(path, arcname=NAME + "/" + rel, recursive=False)
                expected[rel] = (record["sha256"], record["size"])
            for name, content in metadata_files.items():
                rel = "metadata/" + name
                expected[rel] = add_bytes(tar, rel, content)
            for name, content in evidence.items():
                rel = "evidence/" + name
                expected[rel] = add_bytes(tar, rel, content)
            sums = "".join(f"{sha}  {rel}\n" for rel, (sha, _) in sorted(expected.items()))
            expected["metadata/SHA256SUMS.txt"] = add_bytes(tar, "metadata/SHA256SUMS.txt", sums)
        with tarfile.open(temporary, "r:gz") as tar:
            found = {}
            for m in tar:
                if not m.isfile(): raise RuntimeError(f"unexpected tar type: {m.name}")
                rel = m.name.removeprefix(NAME + "/")
                if rel not in expected or m.size != expected[rel][1] or digest_stream(tar.extractfile(m)) != expected[rel][0]:
                    raise RuntimeError(f"runtime archive verification failed: {rel}")
                found[rel] = True
            if set(found) != set(expected): raise RuntimeError("runtime archive member set mismatch")
        archive_sha = digest(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists(): temporary.unlink()
    print(json.dumps({"archive": str(target), "sha256": archive_sha, "bytes": target.stat().st_size, "members": len(expected), "runtime_wheels": len(records), "excluded": len(excluded), "nvdisasm": "included because reachable"}, indent=2))

if __name__ == "__main__": main()
