"""Fail when a repository-relative Markdown link points to a missing file."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")


def main() -> int:
    failures: list[str] = []
    for document in sorted(ROOT.rglob("*.md")):
        if ".git" in document.parts:
            continue
        for target in LINK.findall(document.read_text(encoding="utf-8")):
            target = target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path_text = unquote(target.split("#", 1)[0])
            candidate = (document.parent / path_text).resolve()
            if ROOT not in candidate.parents and candidate != ROOT:
                failures.append(f"{document.relative_to(ROOT)}: escapes root: {target}")
            elif not candidate.exists():
                failures.append(f"{document.relative_to(ROOT)}: missing: {target}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("repository-relative Markdown links: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
