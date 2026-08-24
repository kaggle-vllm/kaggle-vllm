from dataclasses import replace
from hashlib import sha256
from io import BytesIO

from kaggle_vllm.download import download_wheel, immutable_resolve_url
from kaggle_vllm.profiles import load_profile


def small_profile(content: bytes):
    return replace(
        load_profile(),
        wheel_filename="native-cp312.whl",
        wheel_sha256=sha256(content).hexdigest(),
    )


def test_hf_hub_download_is_pinned_and_verified(monkeypatch, tmp_path):
    content = b"wheel-content"
    profile = small_profile(content)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        destination = tmp_path / profile.wheel_filename
        destination.write_bytes(content)
        return str(destination)

    monkeypatch.setattr("kaggle_vllm.download._hub_downloader", lambda: fake_download)
    result = download_wheel(profile, tmp_path)
    assert result.read_bytes() == content
    assert calls == [
        {
            "repo_id": profile.hf_repo_id,
            "filename": profile.wheel_filename,
            "revision": profile.hf_revision,
            "repo_type": "model",
            "local_dir": str(tmp_path.resolve()),
        }
    ]


def test_https_fallback_uses_immutable_resolve_url(monkeypatch, tmp_path):
    content = b"fallback-wheel"
    profile = small_profile(content)
    requests = []

    def opener(request):
        requests.append(request)
        return BytesIO(content)

    monkeypatch.setattr("kaggle_vllm.download._hub_downloader", lambda: None)
    result = download_wheel(profile, tmp_path, opener=opener)
    assert result == (tmp_path / profile.wheel_filename).resolve()
    assert result.read_bytes() == content
    assert requests[0].full_url == immutable_resolve_url(profile)
    assert profile.hf_revision in requests[0].full_url


def test_valid_cached_download_avoids_network(monkeypatch, tmp_path):
    content = b"cached-wheel"
    profile = small_profile(content)
    destination = tmp_path / profile.wheel_filename
    destination.write_bytes(content)
    monkeypatch.setattr(
        "kaggle_vllm.download._hub_downloader",
        lambda: (_ for _ in ()).throw(AssertionError("network lookup not expected")),
    )
    assert download_wheel(profile, tmp_path) == destination.resolve()
