"""CPU audit of the supplementary ALLaM evidence; never imports a GPU runtime."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/kaggle-2026-10-01-allam-7b"
MODEL = "a28dd1e67420cde72d3629c8633a974cf7d9c366"
WHEEL = "5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c"


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected):
    assert math.isfinite(float(actual))
    assert math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10), (
        actual,
        expected,
    )


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def validate_wave(wave):
    requests = wave["requests"]
    assert len(requests) == wave["concurrency"] == wave["request_count"]
    assert wave["success_count"] == len(requests) and wave["error_count"] == 0
    assert wave["success_rate"] == 1 and wave["all_forced_output"]
    for req in requests:
        assert req["ok"] and req["forced_output"] and req["status_code"] == 200
        assert req["text"].strip() and req["completion_tokens"] == 32
        assert req["prompt_tokens"] == 128 and req["total_tokens"] == 160
        assert req["e2e_s"] >= req["ttft_s"] > 0
        close(
            req["tpot_s"],
            (req["e2e_s"] - req["ttft_s"]) / (req["completion_tokens"] - 1),
        )
    prompt = sum(r["prompt_tokens"] for r in requests)
    output = sum(r["completion_tokens"] for r in requests)
    assert (
        wave["total_prompt_tokens"] == prompt and wave["total_output_tokens"] == output
    )
    wall = wave["wave_wall_s"]
    assert wall >= max(r["e2e_s"] for r in requests)
    close(wave["output_tokens_per_s"], output / wall)
    close(wave["total_tokens_per_s"], (prompt + output) / wall)
    close(wave["request_throughput_rps"], len(requests) / wall)
    for metric in ["ttft", "tpot", "e2e"]:
        for p in [50, 95, 99]:
            close(
                wave[f"{metric}_p{p}_s"],
                quantile([r[f"{metric}_s"] for r in requests], p / 100),
            )


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate(root=EVIDENCE, check_hashes=True):
    provenance = read_json(root / "provenance.json")
    for path in root.rglob("*.json"):
        read_json(path)
    for path in root.rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            json.loads(line)
    for item in provenance["payload"]:
        path = root / item["path"]
        assert path.is_file() and path.stat().st_size == item["bytes"]
        assert digest(path) == item["sha256"], path
        assert path.stat().st_size < 10 * 1024 * 1024
        assert path.suffix not in {
            ".pyc",
            ".safetensors",
            ".bin",
            ".pt",
            ".pth",
            ".ckpt",
            ".gguf",
            ".onnx",
            ".whl",
        }
    for link in provenance["notebooks"]:
        source = ROOT / link["source"]
        executed = root / link["executed"]
        assert digest(source) == link["source_sha256"]
        assert digest(executed) == link["executed_sha256"]
        src, exe = read_json(source), read_json(executed)
        assert src["nbformat"] == exe["nbformat"] == 4
        assert len(src["cells"]) == len(exe["cells"])
        for i, (s, e) in enumerate(zip(src["cells"], exe["cells"])):
            expected = "".join(e["source"])
            for change in link["source_changes"]:
                if i == change["cell"]:
                    expected = expected.replace(change["old"], change["new"])
            assert "".join(s["source"]) == expected
            if s["cell_type"] == "code":
                assert not s["outputs"] and s["execution_count"] is None
                assert all(o["output_type"] != "error" for o in e["outputs"])
        # The retained generated runner must occur verbatim in its writer cell.
        runners = list((root / link["stage"]).rglob("*.py"))
        constants = []
        for cell in exe["cells"]:
            if cell["cell_type"] != "code":
                continue
            try:
                tree = ast.parse("".join(cell["source"]))
            except SyntaxError:  # archived notebook shell command
                continue
            constants.extend(
                n.value.strip()
                for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
            )
        assert all(p.read_text().strip() in constants for p in runners)
    for stage in ["p2", "p3", "p4"]:
        runtime = read_json(root / stage / "kaggle-vllm-runtime.json")
        assert runtime["wheel"]["sha256"] == WHEEL
        assert (
            runtime["wheel"]["hf_revision"]
            == "f6b4f10de54924ed6fe9e28cceab84eca7276ab6"
        )
        env = runtime["environment"]
        for key, value in {
            "python": "3.12.13",
            "torch": "2.10.0+cu128",
            "torch_cuda": "12.8",
            "nccl": "2.27.5",
            "driver_version": "580.159.04",
        }.items():
            assert env[key] == value
        assert len(env["gpus"]) == 2
        assert all(
            g["name"] == "Tesla T4" and g["capability"] == [7, 5] for g in env["gpus"]
        )
    p2 = read_json(root / "p2/allam-kaggle-vllm-tp2-result.json")
    assert p2["model_commit"] == MODEL and p2["status"] == "PASS"
    assert p2["tensor_parallel_size"] == 2
    for key in ["prompt", "output"]:
        assert p2[f"total_{key}_tokens"] == sum(
            r[f"{key}_tokens"] for r in p2["outputs"]
        )
    assert all(r["text"].strip() for r in p2["outputs"])
    stages = {"p3": "allam-characterization", "p4": "allam-p4-final"}
    matrices = {}
    for stage, directory in stages.items():
        base = root / stage / directory
        manifest = read_json(base / "evidence-manifest.json")
        assert manifest["model_commit"] == MODEL
        for item in manifest["artifacts"]:
            if "__pycache__/" in item["path"]:
                continue  # explicitly excluded bytecode, identity retained in original manifest
            assert digest(base / item["path"]) == item["sha256"]
        name = "matrix-summary.json" if stage == "p3" else "server-matrix.json"
        matrix = read_json(base / "evidence" / name)
        matrices[stage] = matrix
        assert len(matrix) == 6
        assert {(m["tp"], m["max_model_len"]) for m in matrix} == set(
            itertools.product([1, 2], [960, 2048, 4096])
        )
        for m in matrix:
            expected = (
                "RESOURCE_GATED"
                if m["tp"] == 1
                and m["max_model_len"] >= (2048 if stage == "p3" else 4096)
                else "PASS"
            )
            assert m["status"] == expected
            child = m["child_result" if stage == "p3" else "client_result"]
            log = (
                base
                / "runs"
                / (m["tag"] + (".log" if stage == "p3" else "-server.log"))
            )
            log_text = log.read_text()
            assert MODEL in log_text and "LlamaForCausalLM" in log_text
            assert "0.18.2.dev0+ga26e8dc7f.d20260822" in log_text
            if m["tp"] == 2:
                assert "TP rank 0" in log_text and "TP rank 1" in log_text
            if expected == "RESOURCE_GATED":
                assert child is None and "KV cache" in m["root_cause"]
                continue
            raw = (
                base
                / "runs"
                / (m["tag"] + (".json" if stage == "p3" else "-online.json"))
            )
            assert child == read_json(raw)
            if stage == "p3":
                assert child["model_commit"] == MODEL
                for row in child["benchmarks"]:
                    close(
                        row["output_tokens_per_s"],
                        row["total_output_tokens"] / row["wall_s"],
                    )
                    close(
                        row["total_tokens_per_s"],
                        (row["total_prompt_tokens"] + row["total_output_tokens"])
                        / row["wall_s"],
                    )
                    assert (
                        row["total_output_tokens"]
                        == row["requested_output_tokens_per_request"]
                        * row["request_count"]
                    )
                    assert (
                        row["total_prompt_tokens"]
                        == row["prompt_tokens_per_request"] * row["request_count"]
                    )
            else:
                assert {
                    (w["language"], w["concurrency"], w["repeat"])
                    for w in child["waves"]
                } == set(
                    itertools.product(["english", "arabic"], [1, 2, 4, 8], [1, 2, 3])
                )
                assert len(child["waves"]) == 24
                request_ids = [
                    r["request_id"] for w in child["waves"] for r in w["requests"]
                ]
                assert len(request_ids) == len(set(request_ids)) == 90
                for wave in child["waves"]:
                    validate_wave(wave)
                assert {c["language"] for c in child["capacity_probes"]} == {
                    "english",
                    "arabic",
                }
                assert len(child["capacity_probes"]) == 2
                for cap in child["capacity_probes"]:
                    req = cap["result"]
                    assert req["ok"] and req["completion_tokens"] == 32
                    assert req["prompt_tokens"] == m["max_model_len"] - 96
                    assert req["total_tokens"] == m["max_model_len"] - 64
                assert len(child["chat_sanity"]) == 4 and all(
                    r["ok"] for r in child["chat_sanity"]
                )
    p4 = root / "p4/allam-p4-final/evidence"
    rows = csv_rows(p4 / "final-performance-table.csv")
    assert len(rows) == 40
    for row in rows:
        m = next(m for m in matrices["p4"] if m["tag"] == row["tag"])
        waves = [
            w
            for w in m["client_result"]["waves"]
            if w["language"] == row["language"]
            and w["concurrency"] == int(row["concurrency"])
        ]
        reqs = [r for w in waves for r in w["requests"]]
        assert int(row["waves"]) == 3 and int(row["requests"]) == len(reqs)
        for column, key in [
            ("output_tps", "output_tokens_per_s"),
            ("request_rps", "request_throughput_rps"),
        ]:
            close(row[column + "_mean"], statistics.mean(w[key] for w in waves))
            close(row[column + "_median"], statistics.median(w[key] for w in waves))
        close(
            row["output_tps_std"],
            statistics.stdev(w["output_tokens_per_s"] for w in waves),
        )
        close(
            row["total_tps_mean"],
            statistics.mean(w["total_tokens_per_s"] for w in waves),
        )
        close(row["wave_wall_s_mean"], statistics.mean(w["wave_wall_s"] for w in waves))
        for metric in ["ttft", "tpot", "e2e"]:
            for p in [50, 95, 99]:
                close(
                    row[f"{metric}_p{p}_s"],
                    quantile([r[f"{metric}_s"] for r in reqs], p / 100),
                )
    if check_hashes:
        manifest_paths = set()
        for line in (root / "SHA256SUMS.txt").read_text().splitlines():
            expected, name = line.split("  ", 1)
            assert digest(root / name) == expected, name
            manifest_paths.add(name)
        assert manifest_paths == {
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p.name != "SHA256SUMS.txt"
        }
    return {
        "notebooks": len(provenance["notebooks"]),
        "retained_payload_files": len(provenance["payload"]),
        "p4_configurations": len(matrices["p4"]),
        "p4_waves": sum(
            len((m["client_result"] or {}).get("waves", [])) for m in matrices["p4"]
        ),
        "p4_measured_requests": sum(
            len(w["requests"])
            for m in matrices["p4"]
            for w in (m["client_result"] or {}).get("waves", [])
        ),
        "p4_capacity_probes": sum(
            len((m["client_result"] or {}).get("capacity_probes", []))
            for m in matrices["p4"]
        ),
        "p4_chat_probes": sum(
            len((m["client_result"] or {}).get("chat_sanity", []))
            for m in matrices["p4"]
        ),
        "matched_context_960": [r for r in rows if int(r["max_model_len"]) == 960],
    }


def render_results(result):
    lines = [
        "# Generated ALLaM P4 results",
        "",
        "Matched context 960; throughput is the mean of three wave rates.",
        "Latency is pooled request p50. Units: tok/s and seconds.",
        "",
        "| Language | C | TP | Output tok/s | TTFT p50 | TPOT p50 | E2E p50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        result["matched_context_960"],
        key=lambda r: (r["language"], int(r["concurrency"]), int(r["tp"])),
    ):
        lines.append(
            f"| {row['language']} | {row['concurrency']} | {row['tp']} | "
            f"{float(row['output_tps_mean']):.4f} | {float(row['ttft_p50_s']):.6f} | "
            f"{float(row['tpot_p50_s']):.6f} | {float(row['e2e_p50_s']):.6f} |"
        )
    lines += [
        "",
        "These are descriptive within-server observations, not independent session replicates.",
        "Resource-gated TP1/4096 has no performance row; its throughput is N/A.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args()
    result = validate()
    path = ROOT / "research/allam/summary.json"
    if args.write_summary:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        path.with_name("results.md").write_text(render_results(result))
    else:
        assert read_json(path) == result
        assert path.with_name("results.md").read_text() == render_results(result)
    print(
        "ALLaM evidence: PASS (3 notebooks, 6 P4 configurations, 120 waves, 450 requests)"
    )


if __name__ == "__main__":
    main()
