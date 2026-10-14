"""Target-aware wheel metadata reachability, including requested extras."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .exceptions import BootstrapError


def reachable_distributions(
    requirements: Mapping[str, Sequence[str]],
    roots: Sequence[str],
    *,
    python_version: str = "3.12",
    python_full_version: str = "3.12.13",
    platform_machine: str = "x86_64",
) -> tuple[set[str], list[tuple[str, str, str]]]:
    """Return exact metadata closure and active requirement edges for Linux target."""
    normalized = {canonicalize_name(k): v for k, v in requirements.items()}
    queue = deque((canonicalize_name(root), "") for root in roots)
    visited: set[tuple[str, str]] = set()
    closure: set[str] = set()
    edges: list[tuple[str, str, str]] = []
    while queue:
        name, extra = queue.popleft()
        if (name, extra) in visited:
            continue
        visited.add((name, extra))
        if name not in normalized:
            raise BootstrapError(f"missing runtime distribution: {name}")
        closure.add(name)
        for raw in normalized[name]:
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({"python_version": python_version, "python_full_version": python_full_version, "sys_platform": "linux", "platform_machine": platform_machine, "extra": extra}):
                continue
            child = canonicalize_name(req.name)
            if child not in normalized:
                raise BootstrapError(f"unavailable requirement {name}: {raw}")
            edges.append((name, child, raw))
            for requested_extra in ("", *sorted(req.extras)):
                queue.append((child, requested_extra))
    return closure, edges
