"""Bounded network diagnostics for public Kaggle installation."""
from __future__ import annotations

import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _probe(url: str) -> dict[str, str]:
    host = urlparse(url).hostname
    assert host
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        return {"status": "FAIL", "category": "DNS_FAILURE", "detail": str(error)}
    try:
        with urlopen(Request(url, headers={"User-Agent": "kaggle-vllm/0.3.2"}), timeout=8) as response:
            return {"status": "PASS", "category": "REACHABLE", "detail": str(response.status)}
    except ssl.SSLError as error:
        return {"status": "FAIL", "category": "TLS_ERROR", "detail": str(error)}
    except HTTPError as error:
        category = "PACKAGE_VERSION_UNAVAILABLE" if error.code == 404 and "pypi.org" in host else "HOST_HTTP_ERROR"
        return {"status": "FAIL", "category": category, "detail": str(error.code)}
    except (URLError, TimeoutError, OSError) as error:
        reason = str(error).lower()
        category = "PROXY_FAILURE" if "proxy" in reason else "HOST_UNAVAILABLE"
        return {"status": "FAIL", "category": category, "detail": str(error)}


def network_report() -> dict[str, object]:
    endpoints = {
        "PyPI": "https://pypi.org/pypi/kaggle-vllm/0.3.2/json",
        "Hugging Face": "https://huggingface.co/api/models/waqasm86/kaggle-vllm-binaries",
    }
    probes = {name: _probe(url) for name, url in endpoints.items()}
    try:
        pip = subprocess.run(
            [sys.executable, "-m", "pip", "config", "list"],
            capture_output=True, text=True, timeout=8, check=False,
        )
        config = [line.split("=", 1)[0].strip() for line in pip.stdout.splitlines() if "index" in line.lower()]
    except (OSError, subprocess.SubprocessError):
        config = []
    proxy_present = any(os.environ.get(key) for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"))
    index_host = urlparse(os.environ.get("PIP_INDEX_URL", "")).hostname
    free_bytes = shutil.disk_usage("/tmp").free
    if free_bytes < 24 * 1024**3:
        diagnosis = "DISK_EXHAUSTION"
    elif all(x["category"] == "DNS_FAILURE" for x in probes.values()):
        diagnosis = "DNS_FAILURE_OR_INTERNET_DISABLED"
    elif probes["PyPI"]["category"] == "PACKAGE_VERSION_UNAVAILABLE":
        diagnosis = "PACKAGE_VERSION_UNAVAILABLE"
    elif probes["PyPI"]["status"] != "PASS":
        diagnosis = "PYPI_UNAVAILABLE"
    elif probes["Hugging Face"]["status"] != "PASS":
        diagnosis = "HUGGING_FACE_UNAVAILABLE"
    else:
        diagnosis = "NETWORK_READY"
    return {
        "schema_version": 1,
        "diagnosis": diagnosis,
        "probes": probes,
        "proxy_configured": proxy_present,
        "pip_index_host": index_host,
        "pip_index_configuration": "CUSTOM_INDEX_REVIEW" if index_host and index_host not in ("pypi.org", "pypi.python.org") else "DEFAULT_OR_PYPI",
        "pip_index_config_keys": config,
        "scratch_free_bytes": free_bytes,
        "guidance": "Enable Internet in Kaggle Notebook settings and restart the session if disabled. Check DNS, proxy and pip index settings before retrying. Keep TLS verification enabled.",
    }


def main() -> int:
    report = network_report()
    print(json.dumps(report, indent=2))
    return 0 if report["diagnosis"] == "NETWORK_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
