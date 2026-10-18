"""Read-only validation of the Kaggle CPython 3.12 stack."""

import platform
import sys
from importlib import metadata

EXPECTED = {
    "vllm": "0.30.0",
    "torch": "2.13.0+cu129",
    "triton": "3.7.1",
}

def check_runtime():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Use /usr/bin/python3.12, not the Kaggle "
            f"default interpreter: {sys.executable}"
        )

    if platform.system() != "Linux":
        raise RuntimeError("Linux is required")

    if platform.machine() != "x86_64":
        raise RuntimeError("x86_64 is required")

    versions = {}
    for package, wanted in EXPECTED.items():
        found = metadata.version(package)
        if found != wanted:
            raise RuntimeError(
                f"{package}: expected {wanted}, got {found}"
            )
        versions[package] = found

    return versions

def get_llm_class():
    check_runtime()
    from vllm import LLM
    return LLM
