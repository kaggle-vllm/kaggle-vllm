"""Write evidence-backed 0.3.2 release review reports. No external mutation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/v032-production-release"
STAGING = ROOT / "release/hf-v032-staging"


def md(name: str, body: str) -> None:
    (OUT / name).write_text(body.strip() + "\n")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((OUT / "04-huggingface-existing-inventory.json").read_text())
    verified = json.loads((OUT / "local-artifact-verification.json").read_text())
    licenses = json.loads((OUT / "license-wheel-inventory.json").read_text())
    lock = json.loads((OUT / "07-v032-runtime-manifest.json").read_text())
    native = lock["native"]
    missing = [f"v0.3.2/l4x4/wheelhouse/{x['filename']}" for x in lock["wheels"]]
    missing += [f"v0.3.2/l4x4/native/{native['filename']}"]
    missing += [f"v0.3.2/l4x4/metadata/{x.name}" for x in (STAGING / "v0.3.2/l4x4/metadata").iterdir() if x.is_file()]
    (OUT / "missing-public-l4-artifacts.json").write_text(json.dumps({"count": len(missing), "paths": sorted(missing)}, indent=2) + "\n")
    l4 = next(x for x in verified if x["filename"] == "L4 native")
    t4 = next(x for x in verified if x["filename"] == "T4 native")
    proprietary = [x["filename"] for x in licenses if "proprietary" in x["license_metadata"].lower()]
    unknown = [x["filename"] for x in licenses if not x["license_metadata"]]
    files = sorted((ROOT / "dist").glob("kaggle_vllm-0.3.2*"))
    sums = "\n".join(f"{sha(p)}  {p.name}" for p in files) + "\n"
    (ROOT / "release/v032-SHA256SUMS.txt").write_text(sums)
    src_commit = git("rev-parse", "HEAD")
    md("01-local-repository-audit.md", f"""
# Local repository audit

- Working release branch: `{git('branch', '--show-current')}`; starting commit: `{src_commit}` (2026-10-16).
- The original `kaggle-vllm` main checkout has substantial uncommitted 0.3.0 work and is behind origin/main by 15 commits. It was left untouched. The source branch `fix/v031-offline-base-deps` had an unrelated edit to `competition/paper-track/writeup.md`; it was also left untouched.
- This isolated worktree is based on the newest inspected 0.3.1 line, `fix/v031-offline-base-deps`, which includes the accepted SDK repair, competition gates, GPU evidence and offline-base-deps work.
- The repository includes src, tests, scripts, docs, release, artifacts, research, Kaggle notebooks and GitHub workflows. `dist` in this worktree contains only local 0.3.2 SDK candidates. Historical tags were not changed.
- Python 3.11.15 and pip 26.2.1 were used. No virtual environment, Conda, uv, Ruff, sudo or CUDA toolkit change was used.
""")
    md("02-v020-t4x2-compatibility-audit.md", f"""
# 0.2.0 dual T4 compatibility

The packaged historical profile remains `kaggle-t4x2-cu128`: CPython cp312, two Tesla T4/SM75 GPUs, TP=2, Torch 2.10.0+cu128, CUDA 12.8 and upstream vLLM v0.18.1 at `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`.

The exact wheel `{Path(t4['path']).name}` is {t4['size']} bytes and locally hashed to `{t4['sha256']}` (PASS). The current Hub API reports the same LFS SHA256 at the historical root path. The profile still pins commit `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`. Existing paths and checksum references were not edited.

The older public 0.2.0 package, its Qwen TP=2 sharded-state research and historical acceptance files remain separate from 0.3.2. The new manual notebook has only structural validation; a changed SDK cannot inherit its old GPU acceptance.

The exact PyPI 0.2.0 pure Python wheel was downloaded without installation, hashed against PyPI JSON (SHA256 `f3dce393c9e0bd43b9ba29a29ae14f9467857e5eea61390d41f512a52911fbbe`), and imported in a Python 3.11 subprocess. It reported version 0.2.0 and the T4 profile. This is a CPU packaging check only.
""")
    md("03-v031-to-v032-migration.md", """
# 0.3.1 to 0.3.2 migration

0.3.1 remains published unchanged. The 0.3.2 package changes its version, L4 staging root to `/tmp/kaggle-vllm/v0.3.2/l4x4`, and intended Hub namespace to `v0.3.2/l4x4`. The native vLLM 0.30.0 wheel and 168-wheel lock retain their bytes. The 195-wheel archival closure remains historical.

Public delivery now checks the entire immutable Hub tree against every pinned size and SHA256 before downloading; a resumable cache uses atomic activation, bounded retries and a lock. Runtime staging is serialized by a second lock. Strict CLI bootstrap checks host compatibility before large downloads and returns native diagnostics. `network-preflight` classifies common connectivity failures. The Hub revision is intentionally null until upload and verification.
""")
    md("05-huggingface-reorganization-plan.md", f"""
# Additive Hugging Face reorganization

The current Hub commit is `{inventory['revision']}`, with {inventory['file_count']} files. The T4 wheel remains at the repository root; the L4 native wheel is at `v0.3.0/l4x4/native/`. No wheelhouse is present. The target metadata and release index are staged in `release/hf-v032-staging/`.

1. Complete the third-party redistribution and notice review.
2. Use Hugging Face `CommitOperationCopy` for the two existing LFS wheels, from their immutable source revisions into `v0.2.0/native/` and `v0.3.2/l4x4/native/`. Keep the old paths.
3. Upload the 168 locally verified wheels from `../kaggle-vllm-v030-wheelhouse-cp312-cu129-v2/<group>/` into `v0.3.2/l4x4/wheelhouse/`; upload staged metadata and `releases.json`. No dummy files.
4. Read the resulting immutable commit tree and compare all 170 required L4 entries (168 dependencies, native, lock) with the lock. Download samples or the full set for independent checks as appropriate.
5. Pin the verified commit in the 0.3.2 SDK profile, rebuild SDK distributions, run CPU checks and then the user's manual Kaggle acceptance.

The exact {len(missing)} currently missing target paths are listed in `missing-public-l4-artifacts.json`. Existing paths are retained; deletion requires separate approval. Hugging Face supports server-side copying of LFS files, so the large native wheels need not be downloaded and re-uploaded solely to reorganize them.

Reference: [Hugging Face repository copy documentation](https://huggingface.co/docs/huggingface_hub/guides/repository).
""")
    md("08-native-wheel-verification.md", f"""
# Native wheel verification

- L4: `{Path(l4['path']).name}`, {l4['size']} bytes, SHA256 `{l4['sha256']}`: local PASS. Hub LFS identity at `v0.3.0/l4x4/native/` matches. Source vLLM v0.30.0, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, SM89.
- T4: `{Path(t4['path']).name}`, {t4['size']} bytes, SHA256 `{t4['sha256']}`: local PASS. Historical pinned Hub commit `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`, SM75.

No native wheel was rebuilt. Local Python 3.11 did not import either cp312 wheel. Historical GPU evidence applies to the exact previously tested artifacts only.
""")
    md("09-runtime-wheelhouse-integrity.md", f"""
# L4 runtime wheelhouse integrity

The P33 inference lock has exactly {len(lock['wheels'])} unique dependency wheels totaling {sum(x['size'] for x in lock['wheels']):,} bytes, plus the {native['size']:,}-byte native wheel. The local artifact verifier reported {sum(x['status']=='PASS' for x in verified)} PASS and {sum(x['status']!='PASS' for x in verified)} non-PASS across all 168 dependencies and both native wheels. Each local wheel was checked against the lock's filename, size and SHA256.

The public Hub has zero of the 168 L4 dependency wheels. The exact missing filenames are recorded in `missing-public-l4-artifacts.json`; their source group and hashes are in `07-v032-runtime-manifest.json`. Public delivery is BLOCKED. The SDK does not replace Kaggle base Torch through normal pip dependencies; the locked cu129 Torch remains inside the isolated runtime overlay.
""")
    md("10-sdk-bootstrap-changes.md", """
# SDK bootstrap changes

The CLI accepts `kaggle-vllm bootstrap --profile kaggle-l4x4-cu129 --strict --json`. It checks CPython cp312, Linux x86_64, Kaggle markers and exactly four NVIDIA L4/SM89 GPUs before fetching public data. A local-only 3.11 run can test parsing and integrity but cannot stage cp312 native files.

Public fetching is locked to one validated Hub revision and the packaged 168-wheel allowlist. It validates remote path, size and LFS digest, checks free disk, resumes a partial download with HTTP Range, verifies SHA256, and atomically renames completed files. It serializes cache access and SDK staging. Strict mode runs the native doctor and returns a JSON installation report. Public fetch remains fail closed while the immutable revision is null.

Known limit: Hugging Face delivery still needs a completed public upload and real Kaggle disk/network acceptance. The runtime overlay occupies substantial scratch space; cleanup uses the SDK-owned reset command after reviewing its manifest.
""")
    md("11-kaggle-network-failure-analysis.md", """
# Kaggle network failure

`[Errno -3] Temporary failure in name resolution` is a DNS failure during pip networking. It does not establish any CUDA or wheel defect. The 0.3.2 `network-preflight` probes PyPI and Hugging Face separately with timeouts and reports DNS, TLS, proxy, host, and package-version failures. It also reports whether proxy variables and pip index keys are configured without exposing values.

In Kaggle Notebook settings, enable Internet, restart the session, and rerun the preflight before pip. Pip installation uses bounded retries and timeouts in the manual notebooks. A PyPI 404 before publication is expected. Dependency resolution failures require pip's own error log; disk exhaustion is checked separately by bootstrap. TLS verification and Kaggle network policy remain enabled.
""")
    md("12-local-test-results.md", f"""
# Local Python 3.11 checks

- `PYTHONPATH=src python3.11 -m pytest -q`: 425 passed, 1 skipped (final local CPU suite).
- `python3.11 -m compileall -q src tests`: PASS.
- `python3.11 -m build --no-isolation`: built the SDK wheel and source distribution without creating an environment.
- `python3.11 -m twine check dist/kaggle_vllm-0.3.2*`: PASS.
- Two fixed-epoch wheel builds matched byte-for-byte. Raw setuptools sdists differed in generated timestamps; normalized sdists matched byte-for-byte after `scripts/normalize_sdist_v032.py`. This is a local reproducibility check, not a claim that a later source commit will have the same checksum.
- Wheel metadata: version 0.3.2, Python `>=3.11,<3.13`, pure Python tag, CLI entry point and both profiles present. Sdist contains the new network and downloader modules.
- The exact 0.2.0 PyPI wheel passed Python 3.11 import/profile loading from the wheel archive without installing it.
- Distribution SHA256 values are in `release/v032-SHA256SUMS.txt`.

The first pytest invocation without `PYTHONPATH=src` loaded an older editable install from the dirty main checkout; its 10 failures are an environment path collision, not this worktree. All final tests target this worktree explicitly. Neither cp312 native wheel nor any Kaggle notebook was run locally.
""")
    md("13-license-provenance-audit.md", f"""
# License and provenance audit

All 168 wheel filenames, versions, embedded license metadata and embedded license-file counts are recorded in `license-wheel-inventory.json`. {len(proprietary)} wheels declare NVIDIA proprietary terms, and {len(unknown)} wheels have no usable License metadata. These require package-specific redistribution and notice review before public mirroring. The native vLLM wheel also contains compiled third-party components; its existing publication does not automatically clear a new public mirror.

The SDK is Apache-2.0; that does not grant redistribution rights for NVIDIA CUDA, cuDNN, NCCL, Torch, Triton, FlashAttention, FlashInfer, TorchCodec or every transitive wheel. The local staged NOTICE is a placeholder warning, not final clearance. **Public wheelhouse upload is BLOCKED** until a competent maintainer completes a per-component rights and notice decision. No new binaries were uploaded by this work.

Primary source references: [NVIDIA CUDA license agreement](https://docs.nvidia.com/cuda/eula/), [NCCL source license](https://github.com/NVIDIA/nccl/blob/master/LICENSE.txt), and [vLLM source license](https://github.com/vllm-project/vllm/blob/main/LICENSE). The packaged wheel terms and applicable binary redistribution conditions still need a component-by-component decision.

Proprietary filenames: {', '.join(proprietary)}.

Missing license metadata filenames: {', '.join(unknown)}.
""")
    md("14-github-release-readiness.md", """
# GitHub v0.3.2 release readiness

A local SDK wheel, sdist, SHA256SUMS and release-note draft exist. The native wheel and 6.4 GB dependency wheelhouse should remain on Hugging Face. No tag, GitHub release or PR has been published from this branch yet. A PR should review the source and reports. After the Hub pin and a rebuilt SDK are verified, attach the exact SDK files and checksums to a draft GitHub release. Publish only after the user authorizes that visible action. Mark GPU testing pending until manual evidence arrives.
""")
    md("15-pypi-release-readiness.md", """
# PyPI 0.3.2 readiness

The package name, version, pure Python wheel, source distribution and CLI entry point have local checks. PyPI currently lists 0.3.1 as its latest release; 0.3.2 is not uploaded. The current 0.3.2 build intentionally has an unset Hub revision and cannot deliver a public runtime, so it is a release candidate rather than the final PyPI artifact. Rebuild after the complete immutable Hub pin is committed; verify wheel/sdist byte identity and trusted publishing configuration. Never reuse the previously disclosed token. A PyPI upload requires explicit user authorization.

The proposed `.github/workflows/publish-pypi-v032.yml` is guarded by the reviewed tag, the redistribution clearance file, exact HF completeness, fixed-epoch artifact checksums and a PyPI no-collision check. PyPI's Trusted Publisher configuration must explicitly trust that workflow filename and `pypi` environment before dispatch.
""")
    md("16-huggingface-publication-readiness.md", f"""
# Hugging Face publication readiness

The current Hub tree has {inventory['file_count']} files at `{inventory['revision']}`. Both exact native wheels already exist under historic paths. The target 0.3.2 namespace lacks all 168 dependencies, native copy and metadata. Local metadata is staged in `release/hf-v032-staging/`; all wheel bytes are locally verified, and no large binary was copied into Git.

Publication is BLOCKED by the third-party redistribution review and the user's required authorization. After approval, perform additive copies/uploads, verify the immutable resulting commit and only then pin the SDK. Do not delete root T4 or v0.3.0 L4 paths.
""")
    md("17-manual-kaggle-acceptance-plan.md", """
# Manual Kaggle acceptance

The two output-free notebooks in `kaggle-notebooks/` target the published 0.2.0 T4x2 and proposed 0.3.2 L4x4 releases separately. Both have Internet/DNS preflight, exact GPU checks, PyPI installation, strict bootstrap, native diagnostics, a matching TP generation case, cleanup observation and bounded JSON evidence. The L4 notebook downloads the public, pinned 23.3 GB Gemma 4 QAT model, checks configuration and processor, and repeats semantic TP4 prompts. It needs adequate scratch disk and a complete public runtime.

The notebooks passed JSON and Python syntax validation only. The user must upload and run them in fresh Kaggle sessions. P31 historical Gemma selected `TRITON_ATTN`; FA2 native import is distinct from FA2 model execution. Do not claim throughput or a 0.3.2 GPU PASS until the user's evidence files are reviewed.
""")
    checks = [
        "SDK package integrity", "Python ABI", "GPU profile", "runtime dependency lock",
        "native wheel checksum", "Hugging Face completeness", "immutable revision", "license compliance",
        "downloader correctness", "safe runtime staging", "CPU/static tests", "GitHub release consistency",
        "PyPI artifact consistency", "manual Kaggle SDK install", "manual native imports",
        "manual NCCL tensor parallelism", "manual model generation", "memory cleanup",
        "documentation accuracy", "public distribution reproducibility",
    ]
    t4_status = ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "BLOCKED",
                 "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS",
                 "PASS", "BLOCKED"]
    l4_status = ["PASS", "PASS", "NOT RUN", "PASS", "PASS", "FAIL", "BLOCKED", "BLOCKED",
                 "PASS", "NOT RUN", "PASS", "NOT RUN", "NOT RUN", "NOT RUN", "NOT RUN",
                 "NOT RUN", "NOT RUN", "NOT RUN", "PASS", "BLOCKED"]
    gates = {
        "schema_version": 1, "updated": "2026-10-16",
        "0.2.0": {"scope": "historical exact artifact evidence; no changed SDK acceptance inferred",
                  "overall": "BLOCKED", "checks": dict(zip(checks, t4_status))},
        "0.3.2": {"scope": "local release candidate only", "overall": "BLOCKED",
                  "checks": dict(zip(checks, l4_status))},
    }
    (OUT / "18-production-gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    md("19-publication-action-plan.md", """
# Publication action plan

1. Resolve legal/provenance decisions for the native compiled components and all 168 mirrored wheels; finalize notices.
2. Obtain user authorization for additive Hugging Face publication. Copy the two existing LFS native wheels server-side, upload the verified 168 wheels and staged metadata/index, and preserve old paths.
3. Re-inventory the immutable Hub commit. Fail if any filename, size or SHA256 differs. Pin that commit in the SDK profile and update release metadata.
4. Rebuild 0.3.2 with Python 3.11, run CPU tests, twine and reproducibility checks, and open a source PR for review.
5. Obtain user authorization for PyPI Trusted Publishing and GitHub v0.3.2 release publication. Upload exact reviewed SDK files, checksums and notes. Never upload a credential.
6. The user manually runs both Kaggle notebooks. Review the resulting evidence. Mark 0.3.2 accepted only if all mandatory gates pass.

No remote mutation is performed by this report. The user must explicitly authorize the HF upload, PyPI upload and GitHub release publication as separate visible actions.
""")
    md("20-final-handoff.md", f"""
# 0.3.2 release candidate handoff

The 0.3.2 source and pure Python SDK distributions are prepared in this worktree. The exact 168-wheel local inference closure and both native wheels passed SHA256 checks. The Hub inventory has {inventory['file_count']} files and lacks the public 0.3.2 wheelhouse. The 0.3.2 profile deliberately has no immutable revision; public bootstrap fails clearly. Both manual notebooks are output-free and were not executed.

**Production blockers:** third-party binary redistribution review, approved/additive Hugging Face publication of 168 wheels and metadata, immutable Hub pin, final SDK rebuild, source PR, PyPI/GitHub publication authorization, and fresh manual Kaggle acceptance. No 0.3.2 GPU or Gemma acceptance is claimed.

Historical T4 0.2.0 wheel SHA256: `{t4['sha256']}`. Preserved L4 native wheel SHA256: `{l4['sha256']}`. Local SDK candidate checksums: `release/v032-SHA256SUMS.txt`.
""")
    print("Wrote required release reports and local SDK checksum manifest")


if __name__ == "__main__":
    main()
