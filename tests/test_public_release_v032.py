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


def test_remote_completeness_rejects_missing_wheels(monkeypatch):
    monkeypatch.setattr(public, "_remote_inventory", lambda: {})
    with pytest.raises(BootstrapError, match="missing=.*wheelhouse/"):
        public.verify_remote_completeness()


def test_remote_completeness_verifies_git_stored_small_wheel(monkeypatch):
    lock = public.lean_runtime_manifest()
    namespace = public.load_profile().data["public_artifacts"]["namespace"].strip("/")
    rows = {}
    for item in lock["wheels"]:
        rows[f"{namespace}/wheelhouse/{item['filename']}"] = {
            "size": item["size"], "lfs": {"oid": item["sha256"]}
        }
    native = lock["native"]
    rows[f"{namespace}/native/{native['filename']}"] = {
        "size": native["size"], "lfs": {"oid": native["sha256"]}
    }
    source = public.Path(public.__file__).parent / "profiles/kaggle-l4x4-cu129/L4_INFERENCE_LOCK.json"
    rows[f"{namespace}/metadata/L4_INFERENCE_LOCK.json"] = {"size": source.stat().st_size}
    monkeypatch.setattr(public, "_remote_inventory", lambda: rows)
    monkeypatch.setattr(public, "_remote_small_sha256", lambda relative, size: public.LOCK_SHA256)
    assert public.verify_remote_completeness()["artifact_count"] == 170


def test_network_dns_classification(monkeypatch):
    from kaggle_vllm import network_preflight

    monkeypatch.setattr(network_preflight.socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(network_preflight.socket.gaierror(-3, "DNS unavailable")))
    result = network_preflight._probe("https://pypi.org/pypi/kaggle-vllm/json")
    assert result["category"] == "DNS_FAILURE"
