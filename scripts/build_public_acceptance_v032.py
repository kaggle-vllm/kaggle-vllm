"""Generate output-free manual Kaggle acceptance notebooks; never execute them."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "kaggle-notebooks"


def cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def notebook(title: str, cells: list[dict]) -> dict:
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "cells": [cell("markdown", title), *cells],
    }


PREFLIGHT = r'''
import json, os, pathlib, shutil, socket, subprocess, sys, time, urllib.request
from urllib.error import HTTPError, URLError
EVIDENCE = pathlib.Path("/kaggle/working/kaggle-vllm-public-acceptance.json")
assert sys.version_info[:2] == (3, 12), sys.version
def network(host, path):
    try:
        socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise RuntimeError(f"DNS failure for {host}; enable Internet in Kaggle Notebook settings: {error}") from error
    try:
        with urllib.request.urlopen(f"https://{host}{path}", timeout=12) as response:
            return response.status
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"Network/PyPI/HF failure at {host}; check Internet, proxy, pip index and TLS: {error}") from error
network("pypi.org", PYPI_PATH)
network("huggingface.co", "/api/models/waqasm86/kaggle-vllm-binaries")
gpus = subprocess.check_output(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"], text=True, timeout=15).splitlines()
assert [x.strip() for x in gpus] == EXPECTED_GPUS, gpus
print({"python": sys.version.split()[0], "gpus": gpus, "internet": "PASS"})
'''

T4 = [
    cell("markdown", "Run manually in a fresh Kaggle session with exactly two Tesla T4 GPUs and Internet enabled. This notebook installs the unchanged PyPI 0.2.0 SDK and uses its pinned historical Hugging Face wheel. No Kaggle Input dataset is needed."),
    cell("code", 'PYPI_PATH = "/pypi/kaggle-vllm/0.2.0/json"\nEXPECTED_GPUS = ["Tesla T4, 7.5"] * 2\n' + PREFLIGHT),
    cell("code", r'''
import shlex
subprocess.run([sys.executable, "-m", "pip", "install", "--retries", "3", "--timeout", "20", "kaggle-vllm[hub]==0.2.0"], check=True, timeout=600)
import importlib.metadata
assert importlib.metadata.version("kaggle-vllm") == "0.2.0"
subprocess.run(["kaggle-vllm", "bootstrap", "--strict"], check=True, timeout=1800)
doctor = subprocess.run(["kaggle-vllm", "doctor", "--strict", "--json"], check=True, capture_output=True, text=True, timeout=120)
doctor_data = json.loads(doctor.stdout)
assert doctor_data["compatible"] is True
print("SDK, historical wheel checksum and strict native diagnostics: PASS")
'''),
    cell("code", r'''
from huggingface_hub import snapshot_download
model = snapshot_download(repo_id="facebook/opt-125m", revision="27dcfa74d334bc871f3234de431e71c6eeba5dd6")
before = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
script = f"""from kaggle_vllm import KaggleLLM, KaggleSamplingParams
llm=KaggleLLM({model!r}, profile="kaggle-t4x2-cu128", tensor_parallel_size=2, max_model_len=512, gpu_memory_utilization=0.40)
out=llm.generate(["NCCL enables"], KaggleSamplingParams(temperature=0.0, max_tokens=24))
assert out and out[0].outputs and out[0].outputs[0].text.strip()
print(out[0].outputs[0].text[:200])
"""
command = 'eval "$(kaggle-vllm env)"; exec ' + shlex.quote(sys.executable) + ' -c ' + shlex.quote(script)
result = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=900)
assert result.returncode == 0, result.stderr[-3000:]
time.sleep(10)
after = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
assert all(int(a) <= int(b) + 1024 for a, b in zip(after.strip().splitlines(), before.strip().splitlines())), "GPU memory was not reclaimed"
evidence = {"release": "0.2.0", "status": "PASS", "gpu_profile": "T4x2/SM75", "tensor_parallel": 2,
            "native_sha256": "5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c",
            "hf_revision": "f6b4f10de54924ed6fe9e28cceab84eca7276ab6",
            "generation_excerpt": result.stdout[-200:], "gpu_memory_before": before.strip().splitlines(),
            "gpu_memory_after": after.strip().splitlines(), "doctor": doctor_data}
assert len(json.dumps(evidence)) < 100_000
EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n")
print(EVIDENCE)
'''),
]

L4 = [
    cell("markdown", "Run manually in a fresh Kaggle four L4 GPU session with Internet enabled after the 0.3.2 PyPI release and complete immutable Hugging Face wheelhouse are published. The 23.3 GB Gemma 4 model download is separate from the runtime; check free space first. No Kaggle Input dataset is needed."),
    cell("code", 'PYPI_PATH = "/pypi/kaggle-vllm/0.3.2/json"\nEXPECTED_GPUS = ["NVIDIA L4, 8.9"] * 4\n' + PREFLIGHT),
    cell("code", r'''
import shlex
subprocess.run([sys.executable, "-m", "pip", "install", "--retries", "3", "--timeout", "20", "kaggle-vllm==0.3.2"], check=True, timeout=600)
import importlib.metadata
assert importlib.metadata.version("kaggle-vllm") == "0.3.2"
subprocess.run(["kaggle-vllm", "network-preflight"], check=True, timeout=60)
bootstrap = subprocess.run(["kaggle-vllm", "bootstrap", "--profile", "kaggle-l4x4-cu129", "--strict", "--json"],
                           capture_output=True, text=True, timeout=3600)
assert bootstrap.returncode == 0, bootstrap.stderr[-3000:]
bootstrap_data = json.loads(bootstrap.stdout)
assert bootstrap_data["diagnostics"]["compatible"] is True
print("SDK, 168 wheel hashes, native wheel, strict CUDA diagnostics: PASS")
'''),
    cell("code", r'''
model_repo = "google/gemma-4-31B-it-qat-w4a16-ct"
model_revision = "52f3f65bc7a02d555763bc923bd1d9094898219d"
model_root = pathlib.Path("/tmp/kaggle-vllm/v032-gemma-model")
assert shutil.disk_usage("/tmp").free > 28 * 1024**3, "Need at least 28 GiB free for Gemma 4 model; attach no dataset or use a larger scratch disk."
download_script = f"""from huggingface_hub import snapshot_download
print(snapshot_download(repo_id={model_repo!r}, revision={model_revision!r}, local_dir={str(model_root)!r}, max_workers=4))
"""
command = 'eval "$(kaggle-vllm env --profile kaggle-l4x4-cu129)"; exec ' + shlex.quote(sys.executable) + ' -c ' + shlex.quote(download_script)
download = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=7200)
assert download.returncode == 0, download.stderr[-3000:]
assert (model_root / "config.json").is_file()
print("Pinned Gemma 4 model downloaded to", model_root)
'''),
    cell("code", r'''
before = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
script = f"""import json
from pathlib import Path
from transformers import AutoConfig, AutoProcessor
from kaggle_vllm import KaggleLLM, KaggleSamplingParams
model={str(model_root)!r}
config=AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
processor=AutoProcessor.from_pretrained(model, local_files_only=True, trust_remote_code=False)
assert config and processor
llm=KaggleLLM(model, profile="kaggle-l4x4-cu129", tensor_parallel_size=4, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.86)
rows=[]
for prompt in ("What is 2 + 3 + 5? Reply with the number only.", "What is the length of CUDA? Reply with the number only."):
    for _ in range(2):
        outputs=llm.chat([{{"role":"user","content":prompt}}], sampling_params=KaggleSamplingParams(temperature=0.0,max_tokens=48))
        text=outputs[0].outputs[0].text.strip()
        assert text
        rows.append({{"prompt":prompt,"text":text[:200]}})
assert all("10" in row["text"] for row in rows[:2]), rows[:2]
assert all("4" in row["text"] for row in rows[2:]), rows[2:]
print("KVLLM_EVIDENCE="+json.dumps({{"status":"PASS","rows":rows,"attention_backend":llm.effective_attention_backend()}}, separators=(",",":")))
"""
command = 'eval "$(kaggle-vllm env --profile kaggle-l4x4-cu129)"; exec ' + shlex.quote(sys.executable) + ' -c ' + shlex.quote(script)
result = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, timeout=2400)
assert result.returncode == 0, result.stderr[-3000:]
lines = [line.removeprefix("KVLLM_EVIDENCE=") for line in result.stdout.splitlines() if line.startswith("KVLLM_EVIDENCE=")]
assert len(lines) == 1, result.stdout[-3000:]
generation = json.loads(lines[0])
time.sleep(15)
after = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
assert all(int(a) <= int(b) + 1024 for a, b in zip(after.strip().splitlines(), before.strip().splitlines())), "GPU memory was not reclaimed"
evidence = {"release": "0.3.2", "status": "PASS", "gpu_profile": "L4x4/SM89", "tensor_parallel": 4,
            "native_sha256": "9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8",
            "runtime_revision": bootstrap_data["public_revision"], "bootstrap": bootstrap_data,
            "model": model_repo, "model_revision": model_revision, "generation": generation,
            "gpu_memory_before": before.strip().splitlines(), "gpu_memory_after": after.strip().splitlines()}
assert len(json.dumps(evidence)) < 150_000
EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n")
print(EVIDENCE)
'''),
]


def main() -> None:
    ROOT.mkdir(exist_ok=True)
    for name, title, cells in [
        ("kaggle-vllm-v020-t4x2-public-acceptance.ipynb", "# Manual 0.2.0 dual T4 acceptance", T4),
        ("kaggle-vllm-v032-l4x4-public-acceptance.ipynb", "# Manual 0.3.2 four L4 acceptance", L4),
    ]:
        (ROOT / name).write_text(json.dumps(notebook(title, cells), indent=1) + "\n")
        print(ROOT / name)


if __name__ == "__main__":
    main()
