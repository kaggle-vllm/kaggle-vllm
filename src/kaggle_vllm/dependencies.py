"""Dependency-baseline checks for the separately staged native runtime."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .exceptions import ProfileError
from .profiles import DEFAULT_PROFILE, profile_resource


@dataclass(frozen=True)
class DependencySpec:
    """One distribution required by the validated native-runtime baseline."""

    distribution: str
    import_name: str
    specifier: str
    validated_version: str | None
    required: bool
    source: str


@dataclass(frozen=True)
class DependencyFinding:
    """Serializable result of checking one installed distribution."""

    distribution: str
    import_name: str
    status: str
    installed_version: str | None
    required_specifier: str
    validated_version: str | None
    source: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dependency_baseline(
    profile_name: str = DEFAULT_PROFILE,
) -> tuple[DependencySpec, ...]:
    """Load and validate the curated dependency subset for a profile."""

    resource = profile_resource(profile_name, "dependency-baseline.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ProfileError("dependency baseline schema_version must be 1")
        records = payload["dependencies"]
        if not isinstance(records, list) or not records:
            raise ProfileError("dependency baseline must contain dependencies")
        specs = tuple(
            DependencySpec(
                distribution=str(item["distribution"]),
                import_name=str(item["import_name"]),
                specifier=str(item.get("specifier", "")),
                validated_version=(
                    str(item["validated_version"])
                    if item.get("validated_version") is not None
                    else None
                ),
                required=bool(item.get("required", True)),
                source=str(item["source"]),
            )
            for item in records
        )
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        OSError,
    ) as error:
        raise ProfileError(
            f"unable to load dependency baseline for {profile_name!r}: {error}"
        ) from error

    names = [spec.distribution.casefold() for spec in specs]
    if len(names) != len(set(names)):
        raise ProfileError("dependency baseline distribution names must be unique")
    for spec in specs:
        if not spec.distribution or not spec.import_name or not spec.source:
            raise ProfileError("dependency baseline fields must be non-empty")
        try:
            SpecifierSet(spec.specifier)
            if spec.validated_version is not None:
                Version(spec.validated_version)
        except (InvalidSpecifier, InvalidVersion) as error:
            raise ProfileError(
                f"invalid dependency baseline for {spec.distribution}: {error}"
            ) from error
    return specs


def inspect_dependencies(
    profile_name: str = DEFAULT_PROFILE,
    *,
    strict: bool = False,
    version_lookup: Callable[[str], str] = metadata.version,
) -> tuple[DependencyFinding, ...]:
    """Compare installed distributions with the native wheel/overlay baseline."""

    findings: list[DependencyFinding] = []
    for spec in load_dependency_baseline(profile_name):
        try:
            installed = version_lookup(spec.distribution)
        except metadata.PackageNotFoundError:
            status = "error" if spec.required else "untested"
            findings.append(
                DependencyFinding(
                    spec.distribution,
                    spec.import_name,
                    status,
                    None,
                    spec.specifier,
                    spec.validated_version,
                    spec.source,
                    f"{spec.distribution} is not installed",
                )
            )
            continue

        try:
            parsed = Version(installed)
        except InvalidVersion:
            findings.append(
                DependencyFinding(
                    spec.distribution,
                    spec.import_name,
                    "error",
                    installed,
                    spec.specifier,
                    spec.validated_version,
                    spec.source,
                    f"{spec.distribution} has malformed version metadata: {installed!r}",
                )
            )
            continue

        required = SpecifierSet(spec.specifier)
        if required and parsed not in required:
            status = "error"
            message = (
                f"{spec.distribution} {installed} does not satisfy {spec.specifier}"
            )
        elif spec.validated_version and parsed != Version(spec.validated_version):
            status = "error" if strict else "warning"
            message = (
                f"{spec.distribution} {installed} satisfies the supported range but "
                f"differs from validated {spec.validated_version}"
            )
        else:
            status = "pass"
            message = f"{spec.distribution} {installed}"
        findings.append(
            DependencyFinding(
                spec.distribution,
                spec.import_name,
                status,
                installed,
                spec.specifier,
                spec.validated_version,
                spec.source,
                message,
            )
        )
    return tuple(findings)
