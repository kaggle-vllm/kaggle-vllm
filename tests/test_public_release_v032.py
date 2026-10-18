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
    monkeypatch.setattr(public, "_approved_artifacts", lambda: {"native/test.whl": (3, hashlib.sha256(b"abc").hexdigest())})
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
    monkeypatch.setattr(public, "_approved_artifacts", lambda: {"wheelhouse/test.whl": (6, hashlib.sha256(data).hexdigest())})
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
    monkeypatch.setattr(public, "_approved_artifacts", lambda: {"native/test.whl": (3, hashlib.sha256(b"abc").hexdigest())})
    monkeypatch.setattr(public, "urlopen", lambda *a, **k: Response())
    with pytest.raises(BootstrapError, match="unsafe URL"):
        public._download("native/test.whl", tmp_path / "test.whl", 3, hashlib.sha256(b"abc").hexdigest())


def test_remote_completeness_rejects_missing_metadata_and_native(monkeypatch):
    monkeypatch.setattr(public, "immutable_revision", lambda: "a" * 40)
    monkeypatch.setattr(public, "_remote_inventory", lambda: {})
    with pytest.raises(BootstrapError, match="missing=.*metadata/L4_INFERENCE_LOCK.json"):
        public.verify_remote_completeness()


def test_remote_completeness_verifies_git_stored_metadata(monkeypatch):
    monkeypatch.setattr(public, "immutable_revision", lambda: "a" * 40)
    approved = public._approved_artifacts()
    rows = {}
    for relative, (size, digest) in approved.items():
        rows[public._remote_path(relative)] = {"size": size, "lfs": {"oid": digest}}
    rows[public._remote_path("metadata/L4_INFERENCE_LOCK.json")].pop("lfs")
    monkeypatch.setattr(public, "_remote_inventory", lambda: rows)
    monkeypatch.setattr(public, "_remote_small_sha256", lambda relative, size: public.LOCK_SHA256)
    result = public.verify_remote_completeness()
    assert result["artifact_count"] == 170
    assert result["download_bytes"] == sum(size for size, _ in approved.values())


def test_native_reference_reuses_exact_historical_hub_object():
    lock = public.lean_runtime_manifest()
    remote = public._remote_path("native/" + lock["native"]["filename"])
    assert remote == "v0.3.0/l4x4/native/vllm-0.30.0-cp312-cp312-linux_x86_64.whl"
    assert public.load_profile().wheel_sha256 == lock["native"]["sha256"]


def test_only_manifest_approved_hugging_face_artifacts_can_download(tmp_path):
    approved = public._approved_artifacts()
    assert len(approved) == 170
    with pytest.raises(BootstrapError, match="not approved"):
        public._download("wheelhouse/unlisted.whl", tmp_path / "unlisted.whl", 1, hashlib.sha256(b"x").hexdigest())


def test_network_dns_classification(monkeypatch):
    from kaggle_vllm import network_preflight

    monkeypatch.setattr(network_preflight.socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(network_preflight.socket.gaierror(-3, "DNS unavailable")))
    result = network_preflight._probe("https://huggingface.co/api/models/waqasm86/kaggle-vllm-binaries")
    assert result["category"] == "DNS_FAILURE"
