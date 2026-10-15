"""Fresh Kaggle 4x L4 production-release acceptance runner.

Run only with target CPython 3.12 in a NEW offline Kaggle session. This file
defaults to existing corrected split-v2 Inputs and never downloads artifacts
or builds vLLM. Real-model FA2 inference is outside this competition run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SDK_SHA = "7da66878a85c3b65db18d794d282519d853893f54d7314db5ae551ace377416b"
RUNTIME_SHA = "0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73"
NATIVE_SHA = "9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8"
SDK_NAME = "kaggle_vllm-0.3.0-py3-none-any.whl"
RUNTIME_NAME = "kaggle-vllm-v030-l4x4-runtime-v3.tar.gz"
GEMMA_BASE = Path("/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct")
TARGET = Path("/tmp/kaggle-vllm/v0.3.0/l4x4")
INPUT = Path("/kaggle/input/datasets/waqasm86")
OUT = Path("/kaggle/working/kaggle-vllm-v030-production-acceptance")


def sha256(path: Path, *, progress: bool = False) -> str:
    digest = hashlib.sha256()
    done = 0
    next_notice = 512 * 1024 * 1024
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
            done += len(block)
            if progress and done >= next_notice:
                print(f"  SHA256 progress: {done // (1024 * 1024)} MiB", flush=True)
                next_notice += 512 * 1024 * 1024
    return digest.hexdigest()


def unique_file(root: Path, name: str) -> Path:
    matches = sorted(p for p in root.rglob(name) if p.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(matches)}")
    return matches[0]


def gemma_backend_ranks(log: str) -> dict[int, set[str]]:
    """Read actual worker backend initialization, not a config preference."""
    observed: dict[int, set[str]] = {}
    pattern = re.compile(r"\(Worker_TP([0-3]) pid=\d+\).*Using AttentionBackendEnum\.(\w+) backend")
    for rank, backend in pattern.findall(log):
        observed.setdefault(int(rank), set()).add(backend.upper())
    return observed


def process_group_members(group: int) -> list[int]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            rest = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(rest[2]) == group and rest[0] != "Z":
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def end_group(group: int) -> None:
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and process_group_members(group):
        time.sleep(0.2)
    if process_group_members(group):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output = args.output_root
        self.logs = self.output / "logs"
        self.evidence = self.output / "evidence"
        self.records: list[dict] = []
        self.failed = False
        self.overlay_env = dict(os.environ)
        self.overlay_env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
        self.sdk: Path | None = None
        self.runtime: Path | None = None
        self.native: Path | None = None
        self.components: dict[str, Path] = {}
        self.audit: Path | None = None
        self.gemma_model: Path | None = None

    def child(self, name: str, command: list[str], *, timeout: int, env: dict[str, str] | None = None) -> str:
        """Stream a child log, bound its lifetime, and clean its process group."""
        logfile = self.logs / f"{name}.log"
        print(f"  RUN {name}: {shlex.join(command)}", flush=True)
        lines: list[str] = []
        with logfile.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env or self.overlay_env, start_new_session=True)
            events: queue.Queue[str | None] = queue.Queue()
            def read_stream() -> None:
                assert proc.stdout is not None
                for line in proc.stdout:
                    events.put(line)
                events.put(None)
            reader = threading.Thread(target=read_stream, daemon=True)
            reader.start()
            deadline = time.monotonic() + timeout
            last_heartbeat = time.monotonic()
            eof = False
            try:
                while not eof or proc.poll() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{name} exceeded {timeout}s")
                    try:
                        line = events.get(timeout=0.25)
                    except queue.Empty:
                        if time.monotonic() - last_heartbeat >= 15:
                            heartbeat = f"  {name} still running; elapsed {int(timeout - (deadline - time.monotonic()))}s\n"
                            print(heartbeat, end="", flush=True)
                            log.write(heartbeat)
                            log.flush()
                            last_heartbeat = time.monotonic()
                        continue
                    if line is None:
                        eof = True
                        continue
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    lines.append(line)
                reader.join(timeout=2)
                code = proc.wait(timeout=2)
                if code:
                    raise RuntimeError(f"{name} exited {code}; see {logfile}")
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and process_group_members(proc.pid):
                    time.sleep(0.2)
                remaining = process_group_members(proc.pid)
                if remaining:
                    raise RuntimeError(f"{name} left child processes {remaining}")
                return "".join(lines)
            except BaseException:
                end_group(proc.pid)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    end_group(proc.pid)
                raise

    def gate(self, number: int, name: str, callback, *, required: bool = True) -> None:
        print(f"\nSTEP {number:02d}: {name}", flush=True)
        record = {"step": number, "name": name, "required_for_core": required, "status": "PENDING"}
        if self.failed and number not in (19, 20):
            record["status"] = "SKIPPED_AFTER_REQUIRED_FAILURE"
            self.records.append(record)
            return
        try:
            result = callback()
            if result == "NOT_PROVEN":
                record["status"] = "NOT_PROVEN"
            elif result == "OPTIONAL_NOT_ATTACHED":
                record["status"] = "OPTIONAL"
            else:
                record["status"] = "PASS"
                if isinstance(result, dict):
                    record["evidence"] = result
        except Exception as error:
            record["status"] = "FAIL" if required else "NOT_PROVEN"
            record["error"] = f"{type(error).__name__}: {error}"
            print(f"  {record['status']}: {record['error']}", flush=True)
            if required:
                self.failed = True
        self.records.append(record)

    def inventory(self) -> dict:
        if TARGET.exists():
            raise RuntimeError(f"SDK runtime target already exists; start a genuinely new Kaggle session: {TARGET}")
        self.sdk = self.args.sdk_wheel or unique_file(self.args.input_root, SDK_NAME)
        if not self.sdk.is_file() or Path("/kaggle/input") not in self.sdk.resolve().parents:
            raise RuntimeError(f"SDK wheel must be a mounted Kaggle Input: {self.sdk}")
        roots = sorted(str(p) for p in self.args.input_root.iterdir()) if self.args.input_root.is_dir() else []
        output = self.child("00_nproc", ["nproc", "--all"], timeout=20)
        tmp_space = shutil.disk_usage("/tmp")
        working_space = shutil.disk_usage("/kaggle/working")
        result = {"input_root": str(self.args.input_root), "mounted_entries": roots, "sdk_wheel": str(self.sdk), "runtime_mode": self.args.runtime_mode, "notebook_output_root": str(self.args.notebook_root), "gemma_model_override": str(self.args.gemma_model) if self.args.gemma_model else None, "internet_required": False, "nproc_all": int(output.strip()), "tmp_free_bytes": tmp_space.free, "working_free_bytes": working_space.free}
        if tmp_space.free < 24 * 1024**3:
            raise RuntimeError(f"/tmp has {tmp_space.free} bytes free; SDK requires at least {24 * 1024**3}")
        (self.evidence / "input_inventory.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def _sdk_runtime(self):
        """Read the verified pure-Python SDK wheel before installation."""
        assert self.sdk is not None
        if str(self.sdk) not in sys.path:
            sys.path.insert(0, str(self.sdk))
        from kaggle_vllm import runtime_v3
        return runtime_v3

    def _component_overrides(self) -> dict[str, Path]:
        names = ("runtime-core", "runtime-large", "torch-cu129", "wheelhouse-audit", "build-deps")
        return {name: value for name in names if (value := getattr(self.args, name.replace("-", "_") + "_root", None)) is not None}

    def locate_runtime(self) -> dict:
        if self.args.runtime_mode == "RUNTIME_V3_ARCHIVE":
            self.runtime = self.args.runtime_archive or unique_file(self.args.input_root, RUNTIME_NAME)
            if not self.runtime.is_file() or Path("/kaggle/input") not in self.runtime.resolve().parents:
                raise RuntimeError(f"runtime-v3 must be a mounted Kaggle Input: {self.runtime}")
            return {"mode": self.args.runtime_mode, "runtime_archive": str(self.runtime)}
        if self.args.runtime_archive is not None:
            raise RuntimeError("--runtime-archive requires --runtime-mode RUNTIME_V3_ARCHIVE")
        module = self._sdk_runtime()
        self.components, self.audit = module.discover_split_components(self.args.input_root, overrides=self._component_overrides())
        return {"mode": self.args.runtime_mode, "components": {k: str(v) for k, v in self.components.items()}, "audit": str(self.audit)}

    def locate_native(self) -> dict:
        if self.args.runtime_mode == "RUNTIME_V3_ARCHIVE":
            return {"native_source": "inside runtime-v3"}
        module = self._sdk_runtime()
        explicit = self.args.native_wheel or self.args.p31_archive
        self.native = module.discover_native_source(self.args.input_root, explicit=explicit, notebook_root=self.args.notebook_root)
        if Path("/kaggle/input") not in self.native.resolve().parents:
            raise RuntimeError(f"native source must be a mounted Kaggle Input: {self.native}")
        return {"native_source": str(self.native), "source_kind": "wheel" if self.native.name == module.WHEEL_FILENAME else "P31 final-v2 tar"}

    def validate_inputs(self) -> dict:
        if self.args.runtime_mode == "RUNTIME_V3_ARCHIVE":
            return self.artifact_hash(self.runtime, RUNTIME_SHA, "runtime-v3")
        assert self.native is not None and self.audit is not None
        module = self._sdk_runtime()
        source = module.SplitRuntimeSource(self.args.input_root, self.components, self.native, self.audit)
        result = module.validate_split_source(source, full=True)
        (self.evidence / "05_split_source_integrity.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def hardware(self) -> dict:
        if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12) or sys.platform != "linux" or os.uname().machine != "x86_64":
            raise RuntimeError("requires Kaggle CPython 3.12/Linux x86_64")
        output = self.child("01_nvidia_smi", ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], timeout=60)
        rows = [line.strip() for line in output.splitlines() if line.strip()]
        if len(rows) != 4 or any([part.strip() for part in row.split(",")] != ["NVIDIA L4", "8.9"] for row in rows):
            raise RuntimeError(f"requires exactly 4 NVIDIA L4/SM89 devices: {rows}")
        result = {"python": sys.version.split()[0], "platform": sys.platform, "machine": os.uname().machine, "gpus": rows}
        (self.evidence / "hardware_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def artifact_hash(self, path: Path | None, expected: str, label: str) -> dict:
        assert path is not None
        observed = sha256(path, progress=path.stat().st_size > 1024**3)
        if observed != expected:
            raise RuntimeError(f"{label} SHA256 mismatch: {observed}")
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}

    def install(self) -> None:
        assert self.sdk is not None
        self.artifact_hash(self.sdk, SDK_SHA, "SDK")
        self.child("06_sdk_install", [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-cache-dir", "--force-reinstall", str(self.sdk)], timeout=300)

    def profile(self) -> dict:
        output = self.child("02_profile", ["kaggle-vllm", "profile", "--profile", "kaggle-l4x4-cu129"], timeout=60)
        data = json.loads(output)
        if data["vllm"]["wheel_sha256"] != NATIVE_SHA or data["default_runtime_root"] != str(TARGET):
            raise RuntimeError("installed SDK profile identity mismatch")
        # P33's 168-wheel selection omits these packages. Record where this
        # fresh target actually resolves them before staging the overlay.
        base_dependencies = {}
        for name in ("packaging", "jinja2"):
            try:
                version = importlib.metadata.version(name)
                spec = importlib.util.find_spec(name)
            except (importlib.metadata.PackageNotFoundError, ImportError) as error:
                raise RuntimeError(f"required Kaggle base package {name} is unavailable; the lean P33 lock omits it") from error
            if spec is None or not spec.origin:
                raise RuntimeError(f"required Kaggle base package {name} is not importable")
            base_dependencies[name] = {"version": version, "origin": spec.origin}
        (self.evidence / "02_base_dependencies.json").write_text(json.dumps(base_dependencies, indent=2) + "\n")
        return {"profile": data["name"], "native_sha256": NATIVE_SHA, "base_dependencies": base_dependencies}

    def resolve(self) -> dict:
        runtime = self.locate_runtime()
        native = self.locate_native()
        return {"runtime": runtime, "native": native}

    def bootstrap(self) -> None:
        root = self.runtime if self.args.runtime_mode == "RUNTIME_V3_ARCHIVE" else self.args.input_root
        command = ["kaggle-vllm", "bootstrap", "--profile", "kaggle-l4x4-cu129", "--runtime-root", str(TARGET), "--runtime-source", str(root), "--offline", "--strict", "--json"]
        if self.native is not None:
            command.extend(["--native-wheel", str(self.native)])
        for name, path in self._component_overrides().items():
            command.extend(["--" + name + "-root", str(path)])
        self.child("07_bootstrap", command, timeout=3600)

    def activate(self) -> dict:
        output = self.child("08_env", ["kaggle-vllm", "env", "--profile", "kaggle-l4x4-cu129"], timeout=60)
        names = set()
        for line in output.splitlines():
            if not line.startswith("export "):
                continue
            tokens = shlex.split(line)
            if len(tokens) != 2 or "=" not in tokens[1]:
                raise RuntimeError(f"unexpected SDK environment export: {line}")
            name, value = tokens[1].split("=", 1)
            if name not in ("PYTHONPATH", "LD_LIBRARY_PATH", "LIBRARY_PATH"):
                raise RuntimeError(f"unexpected environment key: {name}")
            self.overlay_env[name] = value
            names.add(name)
        if names != {"PYTHONPATH", "LD_LIBRARY_PATH", "LIBRARY_PATH"} or not self.overlay_env["PYTHONPATH"].startswith(str(TARGET / "pydeps")):
            raise RuntimeError("SDK-owned cu129 overlay is not first on PYTHONPATH")
        result = {"sdk_overlay_first": True, "exported_keys": sorted(names)}
        (self.evidence / "activation.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def probe(self, number: int, name: str) -> dict:
        output = self.evidence / f"{number:02d}_{name.replace('-', '_')}.json"
        self.child(f"{number:02d}_{name}", [sys.executable, str(Path(__file__).with_name("production_target_probe_v030.py")), "--step", name, "--output", str(output)], timeout=300)
        result = json.loads(output.read_text())
        if result.get("status") != "PASS":
            raise RuntimeError(f"target probe {name} did not pass")
        return result

    def doctor(self) -> dict:
        output = self.child("10_doctor", ["kaggle-vllm", "doctor", "--profile", "kaggle-l4x4-cu129", "--strict", "--json", "--import-native"], timeout=900)
        decoder = json.JSONDecoder()
        report = None
        for i, char in enumerate(output):
            if char == "{":
                try:
                    value, _ = decoder.raw_decode(output[i:])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and value.get("schema_version") == 3 and "findings" in value:
                    report = value
                    break
        if report is None or report.get("compatible") is not True:
            raise RuntimeError("strict doctor JSON did not report compatible=true")
        (self.evidence / "10_doctor.json").write_text(json.dumps(report, indent=2) + "\n")
        return {"compatible": True, "check_count": len(report["findings"])}

    def nccl(self) -> dict:
        output = self.child("15_nccl_tp4", [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "-m", "kaggle_vllm.acceptance_v3"], timeout=600)
        ranks = sorted(set(int(x) for x in re.findall(r"PASS rank=(\d+) world_size=4 sum=10(?:\.0)?", output)))
        if ranks != [0, 1, 2, 3]:
            raise RuntimeError(f"NCCL did not report four passing ranks: {ranks}")
        return {"world_size": 4, "passed_ranks": ranks}

    def gemma_discovery(self) -> dict:
        if self.args.gemma_model is None:
            from kaggle_vllm import discover_gemma4_qat
            model = discover_gemma4_qat(GEMMA_BASE)
        else:
            model = self.args.gemma_model
        if not model.is_dir() or not (model / "config.json").is_file() or Path("/kaggle/input") not in model.resolve().parents:
            raise RuntimeError(f"Gemma model must be a mounted offline Kaggle Input: {model}")
        self.gemma_model = model
        return {"model": str(model), "config_sha256": sha256(model / "config.json")}

    def gemma(self) -> dict:
        assert self.gemma_model is not None
        report = self.evidence / "17_gemma.json"
        self.child("17_gemma_semantic", [sys.executable, str(Path(__file__).with_name("fresh_kaggle_gemma_v030.py")), "--model", str(self.gemma_model), "--output", str(report)], timeout=1800)
        data = json.loads(report.read_text())
        if data.get("status") != "PASS" or data.get("meaningful") is not True or data.get("generated_tokens", 0) < 8:
            raise RuntimeError("Gemma wrapper semantic gate did not pass")
        return {"meaningful": True, "generated_tokens": data["generated_tokens"], "backend_diagnostic": data.get("effective_backend_diagnostic")}

    def public_api(self) -> dict:
        self.child("15_public_api", [sys.executable, "-c", "from kaggle_vllm import KaggleLLM, KaggleSamplingParams; print(KaggleLLM.__name__, KaggleSamplingParams.__name__)"], timeout=60)
        return {"exports": ["KaggleLLM", "KaggleSamplingParams"]}

    def backend(self) -> dict:
        data = json.loads((self.evidence / "17_gemma.json").read_text())
        log = (self.logs / "17_gemma_semantic.log").read_text(encoding="utf-8", errors="replace")
        observed = gemma_backend_ranks(log)
        if observed != {rank: {"TRITON_ATTN"} for rank in range(4)}:
            raise RuntimeError(f"Gemma worker runtime backend evidence is incomplete or conflicting: {observed}")
        return {"Gemma_attention_backend": "TRITON_ATTN", "observed_tp_ranks": [0, 1, 2, 3], "wrapper_diagnostic": data.get("effective_backend_diagnostic"), "FA2_extension_separate": True}

    def cleanup(self) -> dict:
        data = json.loads((self.evidence / "17_gemma.json").read_text())
        if data.get("worker_cleanup") != "PASS":
            raise RuntimeError("Gemma TP worker cleanup did not pass")
        return {"worker_cleanup": "PASS", "subprocess_group_cleanup": "PASS"}

    def matrix(self) -> dict:
        required = [row for row in self.records if row["required_for_core"]]
        core = bool(required) and all(row["status"] == "PASS" for row in required)
        passed = {row["step"] for row in self.records if row["status"] == "PASS"}
        result = {"schema_version": 3, "created_utc": datetime.now(timezone.utc).isoformat(), "profile": "kaggle-l4x4-cu129", "runtime_mode": self.args.runtime_mode, "status": "FRESH_KAGGLE_PASS" if core else "FRESH_KAGGLE_FAIL", "FA2_NATIVE_IMPORT_PASS": 11 in passed, "TRITON_ATTN_GEMMA_RUNTIME_PASS": 17 in passed, "GEMMA_TP4_INFERENCE_PASS": 16 in passed, "GEMMA_TP4_CLEANUP_PASS": 18 in passed, "FA2_MODEL_RUNTIME_PASS": False, "fa2_model_status": "NOT_PROVEN_OUTSIDE_COMPETITION", "gates": self.records}
        (self.evidence / "19_final_matrix.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "FA2_MODEL_RUNTIME_PASS": result["FA2_MODEL_RUNTIME_PASS"]}, indent=2), flush=True)
        return {"core_release_pass": core, "fa2_model_runtime_pass": False}

    def bundle(self) -> dict:
        manifest = {"schema_version": 3, "files": []}
        for path in sorted(self.output.rglob("*")):
            if path.is_file() and path.name != "20_evidence_bundle.zip" and path.name != "20_evidence_sha256.json":
                manifest["files"].append({"path": str(path.relative_to(self.output)), "bytes": path.stat().st_size, "sha256": sha256(path)})
        (self.evidence / "20_evidence_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
        archive = self.output / "20_evidence_bundle.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for path in sorted(self.output.rglob("*")):
                if path.is_file() and path != archive:
                    z.write(path, path.relative_to(self.output))
        result = {"bundle": str(archive), "bytes": archive.stat().st_size, "sha256": sha256(archive), "file_count": len(manifest["files"]) + 1}
        print(json.dumps(result, indent=2), flush=True)
        return result

    def cleanup_runtime_after_pass(self, matrix: dict) -> dict:
        """Remove only the SDK-owned runtime after a verified passing bundle."""
        if matrix.get("status") != "FRESH_KAGGLE_PASS":
            raise RuntimeError("runtime cleanup requires FRESH_KAGGLE_PASS")
        bundle = self.records[-1]
        if bundle.get("step") != 20 or bundle.get("status") != "PASS":
            raise RuntimeError("runtime cleanup requires a passing evidence bundle")
        evidence = bundle.get("evidence", {})
        archive = self.output / "20_evidence_bundle.zip"
        if not archive.is_file() or evidence.get("sha256") != sha256(archive):
            raise RuntimeError("runtime cleanup requires a SHA-verified evidence bundle")
        from kaggle_vllm.l4_bootstrap import reset_runtime
        result = reset_runtime(TARGET, yes=True)
        (self.output / "post_pass_runtime_cleanup.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def run(self) -> int:
        self.output.mkdir(parents=True, exist_ok=False)
        self.logs.mkdir()
        self.evidence.mkdir()
        steps = [
            (0, "Python, L4 hardware, storage and nproc inventory", lambda: {**self.inventory(), **self.hardware()}, True),
            (1, "install exact small SDK wheel", self.install, True),
            (2, "select L4x4 profile", self.profile, True),
            (3, "resolve exact offline runtime inputs", self.resolve, True),
            (4, "verify native and selected runtime wheel SHA256", self.validate_inputs, True),
            (5, "bootstrap isolated /tmp runtime", self.bootstrap, True),
            (6, "activate SDK-owned cu129 overlay", self.activate, True),
            (7, "strict doctor", self.doctor, True),
            (8, "Torch 2.13.0+cu129 and CUDA 12.9", lambda: self.probe(8, "torch"), True),
            (9, "upstream vLLM 0.30.0", lambda: self.probe(9, "vllm"), True),
            (10, "native core import", lambda: self.probe(10, "native-core"), True),
            (11, "built-in FA2 native import", lambda: self.probe(11, "native-fa2"), True),
            (12, "real libcuda -lcuda link probe", lambda: self.probe(12, "libcuda-link"), True),
            (13, "NCCL world_size=4", self.nccl, True),
            (14, "offline Gemma model discovery", self.gemma_discovery, True),
            (15, "public SDK API import", self.public_api, True),
            (16, "Gemma SDK wrapper semantic TP4 chat", self.gemma, True),
            (17, "Gemma backend selection", self.backend, True),
            (18, "Gemma TP worker cleanup", self.cleanup, True),
            (19, "final acceptance matrix", self.matrix, False),
            (20, "evidence bundle", self.bundle, False),
        ]
        for number, name, callback, required in steps:
            self.gate(number, name, callback, required=required)
        matrix = json.loads((self.evidence / "19_final_matrix.json").read_text())
        bundle_pass = bool(self.records and self.records[-1]["step"] == 20 and self.records[-1]["status"] == "PASS")
        if matrix["status"] == "FRESH_KAGGLE_PASS" and bundle_pass and self.args.cleanup_runtime_after_pass:
            self.cleanup_runtime_after_pass(matrix)
        return 0 if matrix["status"] == "FRESH_KAGGLE_PASS" and bundle_pass else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=INPUT)
    parser.add_argument("--sdk-wheel", type=Path)
    parser.add_argument("--runtime-mode", choices=("SPLIT_V2_KAGGLE_INPUTS", "RUNTIME_V3_ARCHIVE"), default="SPLIT_V2_KAGGLE_INPUTS")
    parser.add_argument("--runtime-archive", type=Path)
    for name in ("runtime-core", "runtime-large", "torch-cu129", "wheelhouse-audit", "build-deps"):
        parser.add_argument("--" + name + "-root", type=Path)
    native = parser.add_mutually_exclusive_group()
    native.add_argument("--native-wheel", type=Path, help="exact attached P31 wheel")
    native.add_argument("--p31-archive", type=Path, help="exact attached P31 final-v2 tar")
    parser.add_argument("--notebook-root", type=Path, default=Path("/kaggle/input/notebooks"))
    parser.add_argument("--gemma-model", type=Path, help="exact mounted Gemma version; default requires one valid version")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--cleanup-runtime-after-pass", action="store_true", help="remove only the SDK-owned /tmp runtime after a passing, SHA-verified evidence ZIP")
    args = parser.parse_args(argv)
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        parser.error("run this acceptance only on target Kaggle CPython 3.12")
    return Runner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
