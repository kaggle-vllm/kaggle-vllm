"""Ensure tests import the current SDK source checkout."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"

if str(SOURCE) in sys.path:
    sys.path.remove(str(SOURCE))

sys.path.insert(0, str(SOURCE))