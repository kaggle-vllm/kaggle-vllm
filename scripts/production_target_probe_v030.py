"""Target-only CPython 3.12 diagnostics for the fresh L4x4 acceptance runner."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

STEPS = ("torch", "native-wheel", "vllm", "native-core", "native-fa2", "libcuda-link")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("target diagnostics require CPython 3.12")
    from kaggle_vllm import activate_runtime
    if not activate_runtime():
        raise RuntimeError("SDK runtime activation manifest is unavailable")
    result: dict[str, object] = {"schema_version": 3, "step": args.step, "status": "PASS"}
    if args.step == "torch":
        import torch
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        capabilities = [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
        if torch.__version__ != "2.13.0+cu129" or torch.version.cuda != "12.9" or names != ["NVIDIA L4"] * 4 or capabilities != [[8, 9]] * 4 or not torch.cuda.is_available():
            raise RuntimeError(f"target Torch/L4 mismatch: {torch.__version__}, {torch.version.cuda}, {names}, {capabilities}")
        result.update({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "gpu_names": names, "compute_capabilities": capabilities})
    elif args.step == "native-wheel":
        from kaggle_vllm.runtime_v3 import validate_native_wheel
        from kaggle_vllm.profile_v3 import WHEEL_FILENAME, WHEEL_SHA256
        target = Path("/tmp/kaggle-vllm/v0.3.0/l4x4")
        payload = json.loads((target / "manifests" / "sdk-runtime.json").read_text())
        wheel = validate_native_wheel(Path(payload["source"]) / "native" / WHEEL_FILENAME)
        result.update({"native_wheel": str(wheel), "sha256": WHEEL_SHA256})
    elif args.step == "vllm":
        module = importlib.import_module("vllm")
        if module.__version__ != "0.30.0":
            raise RuntimeError(f"wrong upstream vLLM: {module.__version__}")
        result["vllm"] = module.__version__
    elif args.step in ("native-core", "native-fa2"):
        name = "vllm._C_stable_libtorch" if args.step == "native-core" else "vllm.vllm_flash_attn._vllm_fa2_C"
        module = importlib.import_module(name)
        result.update({"module": name, "module_file": getattr(module, "__file__", None)})
    else:
        from kaggle_vllm.cuda_driver import ensure_cuda_driver_linker_alias
        data = ensure_cuda_driver_linker_alias(probe=True)
        if data["status"] != "PASS" or data["link_probe"] != "PASS":
            raise RuntimeError(f"libcuda -lcuda link probe failed: {data}")
        result["linker"] = data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
