"""Safe discovery for the one qualified Kaggle Gemma 4 model family."""
from __future__ import annotations

from pathlib import Path

from .exceptions import RuntimeValidationError

GEMMA4_QAT_BASE = Path("/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct")


def discover_gemma4_qat(base: str | Path = GEMMA4_QAT_BASE) -> Path:
    root = Path(base)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeValidationError(f"Gemma 4 Kaggle Input unavailable: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink() and p.name.isdecimal() and (p / "config.json").is_file()]
    if not candidates:
        raise RuntimeValidationError(f"no valid numbered Gemma 4 model version under {root}")
    latest = max(int(p.name) for p in candidates)
    matches = [p for p in candidates if int(p.name) == latest]
    if len(matches) != 1:
        raise RuntimeValidationError(f"ambiguous latest Gemma 4 model version under {root}")
    return matches[0]
