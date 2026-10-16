"""Check exact locked wheels against publisher indexes without downloading wheels."""

from __future__ import annotations

import concurrent.futures
import hashlib
import html.parser
import json
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "src/kaggle_vllm/profiles/kaggle-l4x4-cu129/L4_INFERENCE_LOCK.json"
OUT = ROOT / "reports/v032-production-release/upstream-wheel-origins.json"
PYTORCH = {"torch", "torchaudio", "torchvision", "torchcodec", "triton"}


class Links(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.values.append(href)


def _get_json(url: str) -> dict:
    with urlopen(Request(url, headers={"User-Agent": "kaggle-vllm-release-audit/0.3.2"}), timeout=25) as response:
        return json.load(response)


def audit(row: dict) -> dict:
    result = {k: row[k] for k in ("distribution", "version", "filename", "sha256", "size")}
    project, version = quote(row["distribution"]), quote(row["version"], safe="")
    pypi = f"https://pypi.org/pypi/{project}/{version}/json"
    try:
        files = _get_json(pypi)["urls"]
        hits = [item for item in files if item["filename"] == row["filename"]]
        if hits and hits[0]["digests"]["sha256"] == row["sha256"] and hits[0]["size"] == row["size"]:
            hit = hits[0]
            result.update({"publisher": "PyPI", "index_url": pypi, "url": hit["url"],
                           "advertised_sha256": hit["digests"]["sha256"], "advertised_size": hit["size"]})
            result["status"] = "EXACT"
            return result
        if hits:
            result["pypi_exact_filename_has_different_digest"] = hits[0]["digests"]["sha256"]
    except Exception as error:
        result["pypi_lookup_error"] = str(error)

    if row["distribution"] in PYTORCH:
        index = f"https://download.pytorch.org/whl/cu129/{project}/"
        publisher = "PyTorch CUDA 12.9 index"
    elif row["distribution"] == "flashinfer-cubin":
        index = "https://flashinfer.ai/whl/flashinfer-cubin/"
        publisher = "FlashInfer index"
    else:
        result["status"] = "NO_MATCHING_PUBLISHER_INDEX"
        return result
    try:
        with urlopen(Request(index, headers={"User-Agent": "kaggle-vllm-release-audit/0.3.2"}), timeout=25) as response:
            links = Links()
            links.feed(response.read().decode("utf-8"))
        matches = [urljoin(index, href) for href in links.values
                   if unquote(urlparse(urljoin(index, href)).path).split("/")[-1] == row["filename"]]
        result.update({"publisher": publisher, "index_url": index})
        if len(matches) != 1:
            result["status"] = "NO_UNIQUE_MATCH"
            return result
        result["url"] = matches[0].split("#", 1)[0]
        result["advertised_sha256"] = parse_qs(urlparse(matches[0]).fragment).get("sha256", [None])[0]
        result["status"] = "EXACT" if result["advertised_sha256"] == row["sha256"] else "HASH_NOT_ADVERTISED" if result["advertised_sha256"] is None else "MISMATCH"
        if result["status"] == "HASH_NOT_ADVERTISED" and row["size"] <= 100_000_000:
            digest = hashlib.sha256()
            seen = 0
            with urlopen(Request(result["url"], headers={"User-Agent": "kaggle-vllm-release-audit/0.3.2"}), timeout=60) as response:
                while chunk := response.read(4 * 1024 * 1024):
                    seen += len(chunk)
                    if seen > row["size"]:
                        break
                    digest.update(chunk)
            result["downloaded_size"] = seen
            result["downloaded_sha256"] = digest.hexdigest() if seen == row["size"] else None
            result["status"] = "EXACT_DOWNLOADED" if seen == row["size"] and digest.hexdigest() == row["sha256"] else "MISMATCH"
        return result
    except Exception as error:
        result["status"] = "INDEX_ERROR"
        result["index_error"] = str(error)
        return result


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(audit, lock["wheels"]))
    counts = {state: sum(row["status"] == state for row in rows)
              for state in sorted({row["status"] for row in rows})}
    payload = {
        "schema_version": 1,
        "purpose": "Read-only publisher-index evidence for hash-pinned retrieval; not redistribution clearance",
        "wheel_count": len(rows),
        "counts": counts,
        "wheels": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, sort_keys=True))
    for row in rows:
        if row["status"] != "EXACT":
            print(row["status"], row["filename"])


if __name__ == "__main__":
    main()
