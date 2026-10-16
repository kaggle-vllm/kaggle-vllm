"""Profile-aware CLI with the historical T4 command path preserved."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Sequence

from .exceptions import KaggleVLLMError
from .l4_bootstrap import DEFAULT_TARGET, bootstrap, reset_runtime, shell_exports
from .profile_v3 import PROFILE_NAME, load_profile
from .profiles import DEFAULT_PROFILE as T4_PROFILE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kaggle-vllm")
    from . import __version__
    parser.add_argument("--version", action="version", version=f"kaggle-vllm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("bootstrap", "plan"):
        b = sub.add_parser(command, help="stage or preview an explicitly selected profile")
        b.add_argument("--profile", choices=(PROFILE_NAME, T4_PROFILE), required=True)
        b.add_argument("--runtime-root", type=Path, default=DEFAULT_TARGET, help="expanded L4 runtime destination")
        b.add_argument("--runtime-source", type=Path, help="offline wheelhouse or runtime source")
        b.add_argument("--native-wheel", type=Path, help="exact local P31 wheel")
        for flag in ("offline", "strict", "dry-run", "reset-runtime", "yes", "json"):
            b.add_argument("--" + flag, action="store_true")
        for flag in ("runtime-core", "runtime-large", "torch-cu129", "wheelhouse-audit", "build-deps"):
            b.add_argument("--" + flag + "-root", type=Path)
    e = sub.add_parser("env")
    e.add_argument("--profile", choices=(PROFILE_NAME, T4_PROFILE), required=True)
    e.add_argument("--runtime-root", type=Path, default=DEFAULT_TARGET)
    d = sub.add_parser("doctor")
    d.add_argument("--profile", choices=(PROFILE_NAME, T4_PROFILE), required=True)
    d.add_argument("--runtime-root", type=Path, default=DEFAULT_TARGET)
    for flag in ("strict", "json", "import-native"):
        d.add_argument("--" + flag, action="store_true")
    p = sub.add_parser("profile")
    p.add_argument("--profile", choices=(PROFILE_NAME, T4_PROFILE))
    g = sub.add_parser("verify-gpus")
    g.add_argument("--profile", choices=(PROFILE_NAME, T4_PROFILE), required=True)
    g.add_argument("--tensor-parallel-size", type=int)
    sub.add_parser("fingerprint")
    sub.add_parser("network-preflight", help="diagnose PyPI and Hugging Face connectivity")
    h = sub.add_parser("verify-hashes", help="verify pinned native wheel and optional staged runtime")
    h.add_argument("--native-wheel", type=Path, required=True)
    h.add_argument("--runtime-source", type=Path)
    m = sub.add_parser("discover-model", help="list local model folders with config.json")
    m.add_argument("--root", type=Path, required=True)
    x = sub.add_parser("evidence-export", help="write a local L4 doctor JSON report")
    x.add_argument("--profile", choices=(PROFILE_NAME,), required=True)
    x.add_argument("--runtime-root", type=Path, default=DEFAULT_TARGET)
    x.add_argument("--output", type=Path, required=True)
    x.add_argument("--strict", action="store_true")
    # Keep the historical entry points visible in top-level help. Their
    # argument handling remains delegated to the original CLI below.
    for command in (
        "build-env", "inspect-shards", "verify-wheel", "stage-wheel",
        "serve", "benchmark", "benchmark-serving", "compare-benchmarks",
    ):
        sub.add_parser(command, help="historical command; see its own --help")
    return parser


def _legacy(argv: list[str]) -> int:
    from .cli import main as legacy_main
    return legacy_main(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import sys
    raw = list(argv if argv is not None else sys.argv[1:])
    if raw and raw[0] not in {"-h", "--help", "--version", "bootstrap", "plan", "env", "doctor", "profile", "verify-gpus", "fingerprint", "network-preflight", "verify-hashes", "discover-model", "evidence-export"}:
        return _legacy(raw)
    parser = build_parser()
    args = parser.parse_args(raw)
    try:
        if args.command == "fingerprint":
            return _legacy(raw)
        if args.command == "network-preflight":
            from .network_preflight import main as preflight_main
            return preflight_main()
        if args.command == "verify-hashes":
            from .runtime_v3 import SplitRuntimeSource, discover_runtime, validate_native_source, validate_split_source, verify_runtime
            result = {"native": validate_native_source(args.native_wheel)}
            if args.runtime_source is not None:
                source = discover_runtime(args.runtime_source, native_source=args.native_wheel)
                result["runtime"] = validate_split_source(source, full=True) if isinstance(source, SplitRuntimeSource) else verify_runtime(source)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "discover-model":
            root = args.root.resolve(strict=True)
            if not root.is_dir():
                parser.error("model root must be a directory")
            matches = []
            for config in root.rglob("config.json"):
                if config.is_symlink() or any(parent.is_symlink() for parent in config.parents if parent != root and root in parent.parents):
                    continue
                if config.is_file() and config.resolve().is_relative_to(root):
                    matches.append(str(config.parent))
                if len(matches) > 100:
                    raise KaggleVLLMError("model discovery exceeded 100 folders; use a narrower root")
            print(json.dumps({"root": str(root), "models": sorted(matches)}, indent=2))
            return 0
        if args.command == "evidence-export":
            from .doctor_v3 import doctor_report
            report = doctor_report(target=args.runtime_root, strict=args.strict)
            output = args.output.expanduser().absolute()
            if output.resolve(strict=False) != output or output.exists() or not output.parent.is_dir():
                raise KaggleVLLMError("evidence output must be a new file in an existing writable directory")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=".kaggle-vllm-evidence-", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(report, handle, indent=2)
                handle.write("\n")
            try:
                os.link(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            print(str(output))
            return 0 if report["compatible"] else 1
        if args.command == "profile":
            if args.profile == T4_PROFILE:
                from .profiles import profile_resource
                print(profile_resource(T4_PROFILE, "profile.json").read_text())
            elif args.profile == PROFILE_NAME:
                print(json.dumps(load_profile().data, indent=2))
            else:
                print(json.dumps({"available_profiles": [T4_PROFILE, PROFILE_NAME], "selection": "explicit"}, indent=2))
            return 0
        if args.profile == T4_PROFILE:
            if args.command in ("bootstrap", "plan"):
                legacy = ["bootstrap", "--profile", T4_PROFILE]
                if args.command == "plan": legacy.append("--dry-run")
                for flag in ("strict", "dry_run", "reset_runtime", "yes", "json"):
                    if getattr(args, flag): legacy.append("--" + flag.replace("_", "-"))
                return _legacy(legacy)
            if args.command == "doctor":
                return _legacy(["doctor"] + (["--strict"] if args.strict else []) + (["--json"] if args.json else []))
            if args.command == "env": return _legacy(["env"])
            return _legacy(["verify-gpus", "--tensor-parallel-size", str(args.tensor_parallel_size or 2)])
        if args.command == "env":
            print("\n".join(shell_exports(args.runtime_root)))
            return 0
        if args.command == "doctor":
            from .doctor_v3 import run_doctor
            return run_doctor(target=args.runtime_root, strict=args.strict, as_json_output=args.json, import_native=args.import_native)
        if args.command == "verify-gpus":
            from .environment import collect
            from .runtime import validate_l4x4_profile
            from .exceptions import RuntimeValidationError
            validate_l4x4_profile(collect())
            if args.tensor_parallel_size not in (None, 4):
                raise RuntimeValidationError("L4x4 profile requires TP=4")
            print("PASS: four NVIDIA L4 / SM89 GPUs")
            return 0
        if args.command == "plan":
            args.dry_run = True
            args.offline = True
        if args.reset_runtime:
            reset = reset_runtime(args.runtime_root, yes=args.yes, dry_run=args.dry_run)
            if args.dry_run:
                print(json.dumps({"reset": reset, "dry_run": True}, indent=2))
                return 0
        overrides = {flag: getattr(args, flag.replace("-", "_") + "_root") for flag in ("runtime-core", "runtime-large", "torch-cu129", "wheelhouse-audit", "build-deps")}
        overrides = {name: path for name, path in overrides.items() if path is not None}
        source = args.runtime_source
        if source is None and not args.offline:
            from .public_artifacts_v3 import fetch_public_runtime, immutable_revision
            from .l4_bootstrap import _compatibility
            problems = _compatibility(args.strict)
            if problems:
                raise KaggleVLLMError("incompatible native runtime: " + "; ".join(problems))
            immutable_revision()
            if args.dry_run:
                print(json.dumps({"profile": PROFILE_NAME, "public_revision": immutable_revision(), "dry_run": True}, indent=2))
                return 0
            source = fetch_public_runtime(args.runtime_root / "download-cache")
        result = bootstrap(runtime_root=source, native_source=args.native_wheel, component_overrides=overrides, target=args.runtime_root, strict=args.strict, dry_run=args.dry_run)
        if source is not None and not args.offline and args.runtime_source is None:
            from .public_artifacts_v3 import immutable_revision
            result["public_revision"] = immutable_revision()
        if args.strict and not args.dry_run:
            from .doctor_v3 import doctor_report
            result["diagnostics"] = doctor_report(target=args.runtime_root, strict=True, import_native=True)
        print(json.dumps(result, indent=2) if args.json else f"{result['profile']}: {'dry run' if args.dry_run else 'staged'} at {result['target']}")
        return 0 if result["compatible"] and result.get("diagnostics", {"compatible": True})["compatible"] else 1
    except KaggleVLLMError as error:
        parser.exit(2, f"kaggle-vllm: {error}\n")
    return 2
