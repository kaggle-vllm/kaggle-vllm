from importlib import metadata

import pytest

from kaggle_vllm.dependencies import DependencySpec, inspect_dependencies


def _inspect(monkeypatch, spec, version=None, *, strict=False):
    monkeypatch.setattr(
        "kaggle_vllm.dependencies.load_dependency_baseline", lambda _profile: (spec,)
    )

    def lookup(name):
        if version is None:
            raise metadata.PackageNotFoundError(name)
        return version

    return inspect_dependencies(strict=strict, version_lookup=lookup)[0]


@pytest.fixture
def required_spec():
    return DependencySpec(
        "example", "example", ">=2,<3", "2.4.0", True, "wheel+overlay"
    )


def test_missing_required_dependency_is_error(monkeypatch, required_spec):
    assert _inspect(monkeypatch, required_spec).status == "error"


def test_missing_optional_dependency_is_untested(monkeypatch, required_spec):
    optional = DependencySpec(
        required_spec.distribution,
        required_spec.import_name,
        required_spec.specifier,
        required_spec.validated_version,
        False,
        required_spec.source,
    )
    assert _inspect(monkeypatch, optional).status == "untested"


def test_out_of_range_dependency_is_error(monkeypatch, required_spec):
    finding = _inspect(monkeypatch, required_spec, "3.0.0")
    assert finding.status == "error"
    assert "does not satisfy" in finding.message


def test_in_range_drift_warns_or_fails_in_strict_mode(monkeypatch, required_spec):
    assert _inspect(monkeypatch, required_spec, "2.5.0").status == "warning"
    assert _inspect(monkeypatch, required_spec, "2.5.0", strict=True).status == "error"


def test_malformed_installed_version_is_error(monkeypatch, required_spec):
    finding = _inspect(monkeypatch, required_spec, "not a version")
    assert finding.status == "error"
    assert "malformed version metadata" in finding.message


def test_t4_triton_profile_does_not_require_flashinfer():
    from kaggle_vllm.dependencies import load_dependency_baseline

    specs = {
        spec.distribution: spec
        for spec in load_dependency_baseline("kaggle-t4x2-cu128")
    }

    assert "flashinfer-python" in specs
    assert specs["flashinfer-python"].required is False
