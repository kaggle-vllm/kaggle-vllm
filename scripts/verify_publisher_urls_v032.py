"""Read-only HEAD availability check of the exact 168 publisher wheel URLs."""

from __future__ import annotations

import concurrent.futures
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/kaggle_vllm/profiles/kaggle-l4x4-cu129/UPSTREAM_WHEEL_SOURCES.json"
OUT = ROOT / "reports/v032-production-release/publisher-availability.json"
ALLOWED = {"files.pythonhosted.org", "download.pytorch.org", "download-r2.pytorch.org"}


def check(row: dict) -> dict:
    result = {"filename": row["filename"], "expected_size": row["size"]}
    for attempt in range(2):
        try:
            request = Request(row["url"], method="HEAD", headers={"User-Agent": "kaggle-vllm-release-audit/0.3.2"})
            with urlopen(request, timeout=30) as response:
                final = urlparse(response.geturl())
                host = final.hostname or ""
                if final.scheme != "https" or not (host in ALLOWED or host == "github.com" or host.endswith(".githubusercontent.com")):
                    raise ValueError(f"untrusted redirect host: {host}")
                length = response.headers.get("Content-Length")
                result.update({"http_status": response.status, "content_length": int(length) if length else None,
                               "final_host": host, "status": "PASS" if response.status == 200 and length and int(length) == row["size"] else "FAIL"})
                return result
        except Exception as error:
            result.update({"status": "FAIL", "error": f"{type(error).__name__}: {error}"})
    return result


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(check, source["wheels"]))
    output = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "wheel_count": len(rows),
              "pass_count": sum(row["status"] == "PASS" for row in rows), "checks": rows}
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"{output['pass_count']}/{output['wheel_count']} exact publisher URLs answered HEAD with the expected Content-Length")
    for row in rows:
        if row["status"] != "PASS":
            print("FAIL", row["filename"], row.get("error", row.get("content_length")))
    if output["pass_count"] != output["wheel_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
