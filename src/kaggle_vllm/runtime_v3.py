"""Offline runtime-v3 discovery and integrity verification."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import zipfile
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from .exceptions import BootstrapError
from .profile_v3 import WHEEL_FILENAME, WHEEL_SHA256

BUNDLE_NAME = "kaggle-vllm-v030-l4x4-runtime-v3"
ARCHIVE_SHA256 = "0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73"
ARCHIVE_SIZE = 6633730070
MANIFEST_SHA256 = "3fbe937af3cc281c4dd1eeadcf1c35fade1c4c2162fca7ac116a942f674a217a"
TORCHCODEC_AUDIT_SHA256 = "e2d7161d69a6214a3de3781eb699e938db52f71546df72ec090f77883e776ce5"
DEFAULT_INPUT_ROOT = Path("/kaggle/input/datasets/waqasm86")
SDK_OWNED_ROOT = Path("/tmp/kaggle-vllm/v0.3.0/l4x4")
P31_ARCHIVE_NAME = "p31-kaggle-vllm-v030-l4x4-final-v2.tar.gz"
P31_ARCHIVE_SHA256 = "e82085d0c6f180d2f4f8eb8f52332a92051273715e0923082fc65fc8857ddad5"
P31_ARCHIVE_SIZE = 291899459
NATIVE_WHEEL_SIZE = 293855397
SPLIT_GROUPS = ("runtime-core", "runtime-large", "torch-cu129", "build-deps")
REQUIRED_DATASET_GROUPS = ("runtime-core", "runtime-large", "torch-cu129", "wheelhouse-audit")
PURE_WHEEL_DIR = "kaggle-vllm-v030-pinned-pure-runtime-wheels"
WHEELHOUSE_MANIFEST_SHA256 = "74e9db98712c7ad8539a30d14d8054e9f52bf71a5e930ede7544b350d2a69e99"
SOURCE_IDENTITY_SHA256 = "ad8fee8ae05618d42d2d215c1de63b78721db661ca552707138c66cc19c24b4c"
SPLIT_ARCHIVE_SHA256 = {
    "runtime-core": "65583967767248a25faeee33515cae87f9ed6324892d441c8f12ef39b8d7cff7",
    "runtime-large": "71b1372e0fc226ac56f1f6e0e775c7a87335f4a5d4802605e8306ebcf22ad6da",
    "torch-cu129": "db05f85e3665fb5aba21a89162a3f06bcb177bb22550340792892c98dcfb4569",
    "wheelhouse-audit": "5d8ea32c662725691f8e9e143d93c629ebd1ec45948096dd9f352a4b96178d1a",
    "build-deps": "db9a3219fde55d16745416d5ea167940bf78ab7d060b3121559484f8da9f0260",
}


@dataclass(frozen=True)
class SplitRuntimeSource:
    root: Path
    components: dict[str, Path]
    native: Path
    audit: Path | None = None


def _pinned_metadata(name: str) -> bytes:
    return files("kaggle_vllm").joinpath("profiles", "kaggle-l4x4-cu129", name).read_bytes()


def pinned_runtime_manifest() -> dict[str, Any]:
    raw = _pinned_metadata("RUNTIME_MANIFEST.json")
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise BootstrapError("packaged runtime manifest SHA256 mismatch")
    return json.loads(raw)


def lean_runtime_manifest() -> dict[str, Any]:
    """The exact 168 dependency wheels selected in the successful P33 run."""
    raw = _pinned_metadata("L4_INFERENCE_LOCK.json")
    data = json.loads(raw)
    if data.get("profile") != "kaggle-l4x4-cu129" or data.get("wheel_count") != 168:
        raise BootstrapError("packaged P33 inference lock is invalid")
    if data.get("native", {}).get("sha256") != WHEEL_SHA256:
        raise BootstrapError("packaged P33 inference lock native identity mismatch")
    names = [record["filename"] for record in data["wheels"]]
    if len(names) != len(set(names)) or any("/" in name or "\\" in name for name in names):
        raise BootstrapError("packaged P33 inference lock has unsafe wheel names")
    return data


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_native_wheel(path: str | Path) -> Path:
    wheel = Path(path)
    if wheel.name != WHEEL_FILENAME or wheel.is_symlink() or not wheel.is_file() or sha256(wheel) != WHEEL_SHA256:
        raise BootstrapError(f"native wheel filename or SHA256 mismatch: {wheel}")
    return wheel


def validate_native_source(path: str | Path) -> dict[str, Any]:
    """Verify a direct wheel or the one pinned P31 tar member without extraction."""
    source = Path(path)
    if source.name == WHEEL_FILENAME:
        validate_native_wheel(source)
        sums = source.parent / "SHA256SUMS.txt"
        if sums.is_file():
            entries = [parts for line in sums.read_text().splitlines() if len(parts := line.split(maxsplit=1)) == 2 and parts[1].lstrip("*") == WHEEL_FILENAME]
            if entries and (len(entries) != 1 or entries[0][0] != WHEEL_SHA256):
                raise BootstrapError("native-wheel checksum manifest mismatch")
        identity = source.parent / "SOURCE_IDENTITY.json"
        if identity.is_file():
            try:
                metadata = json.loads(identity.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise BootstrapError("native-wheel source identity is invalid") from error
            if any(key in metadata for key in ("wheel_filename", "wheel_sha256", "p31_archive_sha256")):
                if metadata.get("wheel_filename") != WHEEL_FILENAME or metadata.get("wheel_sha256") != WHEEL_SHA256 or metadata.get("p31_archive_sha256") != P31_ARCHIVE_SHA256:
                    raise BootstrapError("native-wheel source identity mismatch")
        return {"kind": "wheel", "path": str(source), "sha256": WHEEL_SHA256}
    if source.name != P31_ARCHIVE_NAME or source.stat().st_size != P31_ARCHIVE_SIZE or sha256(source) != P31_ARCHIVE_SHA256:
        raise BootstrapError(f"P31 native source identity mismatch: {source}")
    expected = P31_ARCHIVE_NAME.removesuffix(".tar.gz") + "/" + WHEEL_FILENAME
    with tarfile.open(source, "r:gz") as archive:
        matches = [m for m in archive if m.name == expected]
        if len(matches) != 1 or not matches[0].isfile() or matches[0].size != NATIVE_WHEEL_SIZE:
            raise BootstrapError("P31 tar must contain exactly the pinned regular native wheel member")
        member = archive.extractfile(matches[0])
        assert member is not None
        digest = hashlib.sha256()
        for block in iter(lambda: member.read(4 * 1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != WHEEL_SHA256:
            raise BootstrapError("P31 tar native member SHA256 mismatch")
    return {"kind": "p31-final-v2-tar", "path": str(source), "sha256": WHEEL_SHA256, "archive_sha256": P31_ARCHIVE_SHA256}


def _audit_bytes(component: Path, name: str) -> bytes:
    if component.is_file():
        with zipfile.ZipFile(component) as archive:
            if name not in archive.namelist():
                raise BootstrapError(f"corrected audit ZIP lacks {name}")
            return archive.read(name)
    path = component / name
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(f"corrected audit Dataset lacks {name}")
    return path.read_bytes()


def _wheel_source(component: Path, record: dict[str, Any]) -> tuple[int, str]:
    filename = record["filename"]
    digest = hashlib.sha256()
    count = 0
    if component.is_file():
        with zipfile.ZipFile(component) as archive:
            matches = [info for info in archive.infolist() if info.filename == filename]
            if len(matches) != 1 or matches[0].is_dir() or matches[0].file_size != record["size"]:
                raise BootstrapError(f"missing or unsafe pinned wheel in ZIP: {filename}")
            with archive.open(matches[0]) as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(block)
                    count += len(block)
    else:
        matches = [p for p in component.rglob(filename) if p.is_file() and not p.is_symlink() and component.resolve() in p.resolve().parents]
        if len(matches) != 1:
            raise BootstrapError(f"missing or ambiguous pinned wheel in Dataset: {filename}")
        with matches[0].open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
                count += len(block)
    return count, digest.hexdigest()


def validate_split_source(source: SplitRuntimeSource, *, full: bool = True) -> dict[str, Any]:
    """Cross-check the corrected audit metadata and all selected wheel hashes."""
    if source.audit is None:
        raise BootstrapError("corrected wheelhouse-audit-v2 Dataset is required")
    manifest = lean_runtime_manifest()
    for group in REQUIRED_DATASET_GROUPS:
        component = source.audit if group == "wheelhouse-audit" else source.components[group]
        if component.is_file() and sha256(component) != SPLIT_ARCHIVE_SHA256[group]:
            raise BootstrapError(f"corrected {group} ZIP SHA256 mismatch")
    build_component = source.components.get("build-deps")
    if build_component is not None and build_component.is_file() and sha256(build_component) != SPLIT_ARCHIVE_SHA256["build-deps"]:
        raise BootstrapError("corrected build-deps ZIP SHA256 mismatch")
    audit_raw = _audit_bytes(source.audit, "WHEELHOUSE_MANIFEST.json")
    identity_raw = _audit_bytes(source.audit, "SOURCE_IDENTITY.json")
    if hashlib.sha256(audit_raw).hexdigest() != WHEELHOUSE_MANIFEST_SHA256 or hashlib.sha256(identity_raw).hexdigest() != SOURCE_IDENTITY_SHA256:
        raise BootstrapError("corrected wheelhouse audit metadata identity mismatch")
    audit = json.loads(audit_raw)
    indexed = {item["filename"]: item for item in audit["wheels"]}
    if len(indexed) != audit["wheel_count"]:
        raise BootstrapError("corrected wheelhouse audit has duplicate or missing wheels")
    for record in manifest["wheels"]:
        item = indexed.get(record["filename"])
        if item is None or item["size"] != record["size"] or item["SHA256"] != record["sha256"] or item["package_group"] != record["group"]:
            raise BootstrapError(f"corrected wheelhouse audit disagrees with pinned runtime: {record['filename']}")
        if full:
            size, digest = _wheel_source(source.components[record["group"]], record)
            if size != record["size"] or digest != record["sha256"]:
                raise BootstrapError(f"corrected split wheel SHA256 mismatch: {record['filename']}")
    native = validate_native_source(source.native)
    return {"mode": "SPLIT_V2_KAGGLE_INPUTS", "wheel_count": len(manifest["wheels"]), "audit_manifest_sha256": WHEELHOUSE_MANIFEST_SHA256, "native": native, "full_wheel_hashes_verified": full}


def _component(root: Path, group: str) -> Path:
    slug = f"kaggle-vllm-v030-py312-{group}-v2"
    archives = sorted(p for p in root.rglob(slug + ".zip") if p.is_file())
    marker = "WHEELHOUSE_MANIFEST.json" if group == "wheelhouse-audit" else "*.whl"
    extracted = sorted(p for p in root.rglob(slug) if p.is_dir() and any(p.rglob(marker)))
    # An extracted Dataset may also expose its source ZIP in that same mount.
    # A matching ZIP in another mount is a duplicate corrected Dataset.
    if extracted and any(not any(e in a.parents for e in extracted) for a in archives):
        raise BootstrapError(f"ambiguous corrected v2 {group} Dataset views under {root}")
    choices = extracted if extracted else archives
    if len(choices) != 1:
        raise BootstrapError(f"expected one corrected v2 {group} component under {root}; found {len(choices)}")
    return choices[0]


def discover_native_source(
    dataset_root: str | Path = DEFAULT_INPUT_ROOT,
    *,
    explicit: str | Path | None = None,
    notebook_root: str | Path = Path("/kaggle/input/notebooks"),
) -> Path:
    """Prefer an exact attached P31 Notebook Output; reject ties at each tier."""
    if explicit is not None:
        candidate = Path(explicit)
        if candidate.is_symlink() or not candidate.is_file() or candidate.name not in (WHEEL_FILENAME, P31_ARCHIVE_NAME):
            raise BootstrapError(f"explicit P31 native source must be the pinned wheel or final-v2 tar: {candidate}")
        return candidate
    notebook = Path(notebook_root)
    datasets = Path(dataset_root)
    tiers = [
        sorted(notebook.rglob(WHEEL_FILENAME)) if notebook.is_dir() else [],
        sorted(notebook.rglob(P31_ARCHIVE_NAME)) if notebook.is_dir() else [],
        sorted(datasets.rglob(WHEEL_FILENAME)) if datasets.is_dir() else [],
        sorted(datasets.rglob(P31_ARCHIVE_NAME)) if datasets.is_dir() else [],
    ]
    for matches in tiers:
        regular = [p for p in matches if p.is_file() and not p.is_symlink()]
        if len(regular) > 1:
            raise BootstrapError(f"ambiguous P31 native sources: {regular}")
        if regular:
            return regular[0]
    raise BootstrapError("attach the P31 Notebook Output or a verified dedicated native-wheel Dataset; use --native-source for an exact path")


def discover_split_components(root: str | Path, *, overrides: dict[str, str | Path] | None = None) -> tuple[dict[str, Path], Path]:
    """Find exact corrected v2 Dataset views without selecting a native source."""
    root = Path(root)
    overrides = overrides or {}
    if set(overrides) - set(REQUIRED_DATASET_GROUPS) - {"build-deps"}:
        raise BootstrapError("unknown corrected split component override")
    def select(group: str) -> Path:
        if group not in overrides:
            return _component(root, group)
        path = Path(overrides[group])
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise BootstrapError(f"invalid explicit corrected {group} component: {path}")
        if path.is_dir():
            slug = f"kaggle-vllm-v030-py312-{group}-v2"
            marker = "WHEELHOUSE_MANIFEST.json" if group == "wheelhouse-audit" else "*.whl"
            if path.name == slug and any(path.rglob(marker)):
                return path
            if group == "build-deps" and path.name == PURE_WHEEL_DIR and any(path.glob("*.whl")):
                return path
            return _component(path, group)
        return path
    components = {group: select(group) for group in REQUIRED_DATASET_GROUPS if group != "wheelhouse-audit"}
    audit = select("wheelhouse-audit")
    # P33's lean inference lock contains no build-deps wheels. Retain discovery
    # of an explicitly attached historical component, but never require it.
    try:
        components["build-deps"] = select("build-deps")
    except BootstrapError as error:
        if "found 0" not in str(error):
            raise
        pure = sorted(p for p in root.rglob(PURE_WHEEL_DIR) if p.is_dir())
        if len(pure) > 1:
            raise BootstrapError(f"ambiguous optional {PURE_WHEEL_DIR} views under {root}")
        if pure:
            components["build-deps"] = pure[0]
    return components, audit


def discover_runtime(
    source_root: str | Path | None = None,
    *,
    native_source: str | Path | None = None,
    notebook_root: str | Path = Path("/kaggle/input/notebooks"),
    component_overrides: dict[str, str | Path] | None = None,
) -> Path | SplitRuntimeSource:
    """Prefer corrected split-v2 inputs; accept runtime-v3 when explicitly supplied."""
    root = Path(source_root) if source_root is not None else DEFAULT_INPUT_ROOT
    if root.is_file() and root.name == BUNDLE_NAME + ".tar.gz":
        return root
    if not root.is_dir():
        raise BootstrapError(f"runtime input does not exist: {root}")
    if (root / "metadata" / "RUNTIME_MANIFEST.json").is_file() and (root / "native" / WHEEL_FILENAME).is_file():
        return root
    has_split = bool(component_overrides) or any(root.rglob("kaggle-vllm-v030-py312-*-v2.zip")) or any(p.is_dir() for p in root.rglob("kaggle-vllm-v030-py312-*-v2"))
    if has_split:
        components, audit = discover_split_components(root, overrides=component_overrides)
        native = discover_native_source(root, explicit=native_source, notebook_root=notebook_root)
        return SplitRuntimeSource(root, components, native, audit)
    archives = set(root.rglob(BUNDLE_NAME + ".tar.gz"))
    manifests = {p.parent.parent for p in root.rglob("RUNTIME_MANIFEST.json") if (p.parent.parent / "native" / WHEEL_FILENAME).is_file()}
    unique = sorted(archives | manifests)
    if len(unique) != 1:
        raise BootstrapError(f"expected one runtime-v3 artifact under {root}; found {len(unique)}")
    return unique[0]


def _owned_destination(destination: Path) -> Path:
    lexical = Path(os.path.abspath(destination.expanduser()))
    if lexical != destination.expanduser().resolve(strict=False):
        raise BootstrapError(f"runtime staging cannot traverse symlinks: {destination}")
    if lexical == Path("/") or lexical == Path("/tmp") or Path("/kaggle/working") == lexical or Path("/kaggle/working") in lexical.parents:
        raise BootstrapError(f"unsafe runtime staging destination: {destination}")
    owned = lexical == SDK_OWNED_ROOT or SDK_OWNED_ROOT in lexical.parents
    forbidden_parents = (Path("/kaggle/input"), Path("/etc"), Path("/usr"), Path("/var"), Path("/bin"), Path("/sbin"), Path("/lib"), Path("/opt"))
    custom = "kaggle-vllm-v030-runtime" in lexical.parts and all(lexical != p and p not in lexical.parents for p in forbidden_parents)
    if not (owned or custom):
        raise BootstrapError(f"custom runtime root must contain an SDK-owned kaggle-vllm-v030-runtime directory: {destination}")
    return lexical


def _safe_member(name: str, prefix: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts and len(path.parts) >= 2 and path.parts[0] == prefix


def extract_runtime(archive: Path, destination: Path, *, dry_run: bool = False) -> Path:
    """Read every tar header and reject links/traversal before extraction."""
    if dry_run:
        return destination / BUNDLE_NAME
    destination = _owned_destination(destination)
    if archive.stat().st_size != ARCHIVE_SIZE or sha256(archive) != ARCHIVE_SHA256:
        raise BootstrapError(f"runtime-v3 archive identity mismatch: {archive}")
    if destination.exists() and any(destination.iterdir()):
        raise BootstrapError(f"refusing nonempty runtime extraction target: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        seen: set[str] = set()
        for member in members:
            if not member.isfile() or not _safe_member(member.name, BUNDLE_NAME) or member.name in seen:
                raise BootstrapError(f"unsafe runtime archive member: {member.name}")
            seen.add(member.name)
        tar.extractall(destination, members=members, filter="data")
    return destination / BUNDLE_NAME


def stage_split_runtime(source: SplitRuntimeSource, destination: Path) -> Path:
    """Copy only pinned wheels from corrected v2 datasets, without archive extraction."""
    destination = _owned_destination(destination)
    if destination.exists() and any(destination.iterdir()):
        raise BootstrapError(f"refusing nonempty split staging target: {destination}")
    validate_split_source(source, full=True)
    manifest = lean_runtime_manifest()
    destination.mkdir(parents=True)
    wheelhouse = destination / "wheelhouse"
    wheelhouse.mkdir()
    for record in manifest["wheels"]:
        group = record["group"]
        component = source.components[group]
        target = wheelhouse / record["filename"]
        if component.is_file():
            with zipfile.ZipFile(component) as archive:
                entries = [x for x in archive.infolist() if Path(x.filename).name == target.name]
                if len(entries) != 1 or entries[0].filename != target.name or entries[0].is_dir() or entries[0].file_size != record["size"]:
                    raise BootstrapError(f"split component wheel missing or unsafe: {target.name}")
                with archive.open(entries[0]) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
        else:
            matches = list(component.rglob(target.name))
            if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_file():
                raise BootstrapError(f"split component wheel missing or ambiguous: {target.name}")
            shutil.copyfile(matches[0], target)
        if target.stat().st_size != record["size"] or sha256(target) != record["sha256"]:
            raise BootstrapError(f"split component wheel identity mismatch: {target.name}")
    native_dir = destination / "native"
    native_dir.mkdir()
    native_target = native_dir / WHEEL_FILENAME
    if source.native.name == WHEEL_FILENAME:
        shutil.copyfile(validate_native_wheel(source.native), native_target)
    else:
        with tarfile.open(source.native, "r:gz") as archive:
            exact = P31_ARCHIVE_NAME.removesuffix(".tar.gz") + "/" + WHEEL_FILENAME
            matches = [m for m in archive if m.name == exact]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size != NATIVE_WHEEL_SIZE:
                raise BootstrapError("P31 native wheel member identity mismatch")
            with archive.extractfile(matches[0]) as inp, native_target.open("wb") as out:
                shutil.copyfileobj(inp, out)
    validate_native_wheel(native_target)
    meta = destination / "metadata"
    meta.mkdir()
    (meta / "RUNTIME_MANIFEST.json").write_bytes(_pinned_metadata("L4_INFERENCE_LOCK.json"))
    audit_bytes = _pinned_metadata("TORCHCODEC_ELF_AUDIT.json")
    if hashlib.sha256(audit_bytes).hexdigest() != TORCHCODEC_AUDIT_SHA256:
        raise BootstrapError("packaged TorchCodec ELF audit SHA256 mismatch")
    (meta / "TORCHCODEC_ELF_AUDIT.json").write_bytes(audit_bytes)
    verify_runtime(destination)
    return destination


def load_runtime_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "metadata" / "RUNTIME_MANIFEST.json"
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() not in (MANIFEST_SHA256, hashlib.sha256(_pinned_metadata("L4_INFERENCE_LOCK.json")).hexdigest()):
            raise BootstrapError("runtime manifest SHA256 mismatch")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid runtime manifest: {error}") from error
    if data.get("schema_version") != 3 or data.get("profile") != "kaggle-l4x4-cu129" or data.get("native", {}).get("filename") != WHEEL_FILENAME or data.get("native", {}).get("sha256") != WHEEL_SHA256:
        raise BootstrapError("runtime manifest identity mismatch")
    records = data.get("wheels")
    if not isinstance(records, list) or not records:
        raise BootstrapError("runtime manifest has no wheel closure")
    names = [x.get("filename") for x in records]
    if len(names) != len(set(names)) or any(not isinstance(x, str) or "/" in x or "\\" in x for x in names):
        raise BootstrapError("runtime manifest wheel paths are invalid")
    if data.get("wheel_count") != len(records):
        raise BootstrapError("runtime manifest wheel count mismatch")
    for record in records:
        if not isinstance(record.get("size"), int) or record["size"] < 1 or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))):
            raise BootstrapError("runtime manifest wheel size or SHA256 is invalid")
        if not record.get("distribution") or not record.get("version"):
            raise BootstrapError("runtime manifest wheel identity is incomplete")
    return data


def verify_runtime(root: str | Path, *, full: bool = True) -> dict[str, Any]:
    root = Path(root)
    data = load_runtime_manifest(root)
    validate_native_wheel(root / "native" / WHEEL_FILENAME)
    if full:
        for record in data["wheels"]:
            path = root / "wheelhouse" / record["filename"]
            if not path.is_file() or path.stat().st_size != record["size"] or sha256(path) != record["sha256"]:
                raise BootstrapError(f"runtime wheel integrity mismatch: {path}")
    if any("torchcodec-0.17" in x["filename"].lower() for x in data["wheels"]):
        raise BootstrapError("old TorchCodec 0.17 wheel is forbidden")
    if not any(x["distribution"] == "torchcodec" and x["version"] == "0.16.0+cu129" for x in data["wheels"]):
        raise BootstrapError("corrected TorchCodec is missing")
    return data
