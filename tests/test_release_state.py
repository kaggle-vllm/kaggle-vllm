"""Release checks must reuse exact PyPI bytes and fail closed on collisions."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "release_state.py"
spec = importlib.util.spec_from_file_location("release_state", SCRIPT)
release_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_state)


def test_classify_exact_and_collision():
    lock = release_state.load_lock()
    urls = [
        {"filename": item["filename"], "size": item["size"],
         "digests": {"sha256": item["sha256"]}}
        for item in lock["files"]
    ]
    assert release_state.classify_pypi(
        {"info": {"version": lock["version"]}, "urls": urls}, lock
    ) == "SDK_RELEASE_COMPLETE"
    changed = [dict(item) for item in urls]
    changed[0] = {**changed[0], "digests": {"sha256": "0" * 64}}
    with pytest.raises(ValueError, match="collision"):
        release_state.classify_pypi(
            {"info": {"version": lock["version"]}, "urls": changed}, lock
        )


def test_verify_dir_rejects_extra_and_symlink(tmp_path):
    files = []
    for name, content in (("sdk.whl", b"sdk"), ("sdk.tar.gz", b"source")):
        path = tmp_path / name
        path.write_bytes(content)
        files.append({"filename": name, "size": len(content),
                      "sha256": hashlib.sha256(content).hexdigest()})
    lock = {"files": files}
    release_state.verify_dir(tmp_path, lock)
    extra = tmp_path / "unexpected"
    extra.write_bytes(b"x")
    with pytest.raises(ValueError, match="filename set mismatch"):
        release_state.verify_dir(tmp_path, lock)
    extra.unlink()
    (tmp_path / "sdk.whl").unlink()
    (tmp_path / "sdk.whl").symlink_to(tmp_path / "sdk.tar.gz")
    with pytest.raises(ValueError, match="filename set mismatch"):
        release_state.verify_dir(tmp_path, lock)
