"""Generate a small per-wheel rationale bound to immutable runtime-v3 metadata."""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "src/kaggle_vllm/profiles/kaggle-l4x4-cu129"
RELEASE = ROOT / "release"
EXPECTED_MANIFEST_SHA = "3fbe937af3cc281c4dd1eeadcf1c35fade1c4c2162fca7ac116a942f674a217a"


def main() -> None:
    raw = (PROFILE / "RUNTIME_MANIFEST.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("pinned manifest identity mismatch")
    manifest = json.loads(raw)
    graph = json.loads((RELEASE / "RUNTIME_DEPENDENCY_GRAPH.json").read_text())
    excluded = json.loads((RELEASE / "RUNTIME_EXCLUSIONS.json").read_text())
    roots = graph["roots"]
    children: dict[str, list[dict]] = {}
    for edge in graph["edges"]:
        children.setdefault(edge["from"], []).append(edge)
    parents: dict[str, dict | None] = {root: None for root in roots}
    queue = deque(roots)
    while queue:
        parent = queue.popleft()
        for edge in children.get(parent, []):
            child = edge["to"]
            if child not in parents:
                parents[child] = edge
                queue.append(child)
    records = []
    for wheel in manifest["wheels"]:
        name = wheel["distribution"]
        if name not in parents:
            raise RuntimeError(f"unreachable selected wheel: {name}")
        path = [name]
        reasons = []
        while (edge := parents[path[-1]]) is not None:
            reasons.append(edge["requirement"])
            path.append(edge["from"])
        path.reverse()
        reasons.reverse()
        records.append({"distribution": name, "version": wheel["version"], "filename": wheel["filename"], "group": wheel["group"], "root": path[0], "root_path": path, "active_requires_dist": reasons})
    if len(records) != 195 or graph["reachable_count_including_native"] != 196 or len(excluded["excluded"]) != 8:
        raise RuntimeError("runtime closure count changed")
    result = {"schema_version": 3, "runtime_manifest_sha256": EXPECTED_MANIFEST_SHA, "runtime_archive_sha256": "0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73", "target": "CPython 3.12/Linux x86_64", "roots": roots, "selected_wheel_count": len(records), "excluded_count": len(excluded["excluded"]), "packages": records}
    output = RELEASE / "RUNTIME_PACKAGE_RATIONALE.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output, len(records), "selected wheels")


if __name__ == "__main__":
    main()
