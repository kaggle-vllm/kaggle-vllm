"""CPU-only checks for the 0.3.2 public delivery boundary."""
from __future__ import annotations

import hashlib
import io

import pytest

from kaggle_vllm.exceptions import BootstrapError
from kaggle_vllm import public_artifacts_v3 as public


def test_public_delivery_remains_unpinned_until_complete():
    with pytest.raises(BootstrapError, match="immutable"):
        public.immutable_revision()


def test_downloader_rejects_symlink_cache(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.symlink_to(tmp_path / "elsewhere")
    monkeypatch.setattr(public, "_artifact_url", lambda _: "https://huggingface.co/pinned")
    with pytest.raises(BootstrapError, match="symlink"):
        public._download("native/test.whl", target, 3, hashlib.sha256(b"abc").hexdigest())


def test_downloader_resumes_range_and_checks_hash(tmp_path, monkeypatch):
    data = b"abcdef"
    target = tmp_path / "test.whl"
    target.with_name("test.whl.part").write_bytes(b"abc")
    requests = []

    class Response:
        status = 206
        headers = {"Content-Range": "bytes 3-5/6"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            return self.buffer.read(size)

        buffer = io.BytesIO(b"def")

    def open_request(request, timeout):
        requests.append(request.get_header("Range"))
        return Response()

    monkeypatch.setattr(public, "_artifact_url", lambda _: "https://huggingface.co/pinned")
    monkeypatch.setattr(public, "urlopen", open_request)
    public._download("wheelhouse/test.whl", target, 6, hashlib.sha256(data).hexdigest())
    assert requests == ["bytes=3-"]
    assert target.read_bytes() == data
    assert not target.with_name("test.whl.part").exists()


def test_hub_downloader_rejects_insecure_redirect(tmp_path, monkeypatch):
    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self):
            return "http://cas-bridge.xethub.hf.co/wheel"

    monkeypatch.setattr(public, "_artifact_url", lambda _: "https://huggingface.co/pinned")
    monkeypatch.setattr(public, "urlopen", lambda *a, **k: Response())
    with pytest.raises(BootstrapError, match="unsafe URL"):
        public._download("native/test.whl", tmp_path / "test.whl", 3, hashlib.sha256(b"abc").hexdigest())


def test_remote_completeness_rejects_missing_metadata_and_native(monkeypatch):
    monkeypatch.setattr(public, "_remote_inventory", lambda: {})
    with pytest.raises(BootstrapError, match="missing=.*metadata/L4_INFERENCE_LOCK.json"):
        public.verify_remote_completeness()


def test_remote_completeness_verifies_git_stored_metadata(monkeypatch):
    lock = public.lean_runtime_manifest()
    namespace = public.load_profile().data["public_artifacts"]["namespace"].strip("/")
    rows = {}
    native = lock["native"]
    rows[public._remote_path(f"native/{native['filename']}")] = {
        "size": native["size"], "lfs": {"oid": native["sha256"]}
    }
    source = public.Path(public.__file__).parent / "profiles/kaggle-l4x4-cu129/L4_INFERENCE_LOCK.json"
    rows[f"{namespace}/metadata/L4_INFERENCE_LOCK.json"] = {"size": source.stat().st_size}
    source_map = source.with_name("UPSTREAM_WHEEL_SOURCES.json")
    rows[f"{namespace}/metadata/UPSTREAM_WHEEL_SOURCES.json"] = {"size": source_map.stat().st_size}
    monkeypatch.setattr(public, "_remote_inventory", lambda: rows)
    monkeypatch.setattr(public, "_remote_small_sha256", lambda relative, size: public.LOCK_SHA256 if relative.endswith("L4_INFERENCE_LOCK.json") else public.UPSTREAM_SOURCES_SHA256)
    result = public.verify_remote_completeness()
    assert result["artifact_count"] == 3
    assert result["download_bytes"] >= sum(item["size"] for item in lock["wheels"])


def test_native_reference_reuses_exact_historical_hub_object():
    lock = public.lean_runtime_manifest()
    remote = public._remote_path("native/" + lock["native"]["filename"])
    assert remote == "v0.3.0/l4x4/native/vllm-0.30.0-cp312-cp312-linux_x86_64.whl"
    assert public.load_profile().wheel_sha256 == lock["native"]["sha256"]


def test_publisher_sources_match_full_lock_and_pytorch_triton():
    sources = public._publisher_sources(public.lean_runtime_manifest())
    assert len(sources) == 168
    assert sources["triton-3.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"].startswith("https://download-r2.pytorch.org/")


def test_network_dns_classification(monkeypatch):
    from kaggle_vllm import network_preflight

    monkeypatch.setattr(network_preflight.socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(network_preflight.socket.gaierror(-3, "DNS unavailable")))
    result = network_preflight._probe("https://pypi.org/pypi/kaggle-vllm/json")
    assert result["category"] == "DNS_FAILURE"
