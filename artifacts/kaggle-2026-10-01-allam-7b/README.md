# ALLaM-7B supplementary evidence, 2026-10-01

Status: **REVIEWED_SUPPLEMENTARY_EVIDENCE**, distinct from frozen M1–M4.
See the [case-study report](../../research/allam/README.md) for protocols,
results, discrepancy review, paper citation and limitations.

## Retained sources

| Stage | Output-free source | Executed code / code / markdown cells | Jupyter errors |
|---|---|---|---:|
| P2 compatibility | [source](../../kaggle-notebooks/allam-7b-instruct-kaggle-vllm.ipynb) | 9 / 10 / 8 | 0 |
| P3 offline exploration | [source](../../kaggle-notebooks/p3-allam-7b-kaggle-vllm-characterization.ipynb) | 17 / 18 / 16 | 0 |
| P4 online serving | [source](../../kaggle-notebooks/p4-allam-7b-kaggle-vllm-final-research.ipynb) | 16 / 17 / 15 | 0 |

`p2/`, `p3/` and `p4/` preserve 87 files byte-for-byte from the 89-file ALLaM
archive, including all three executed notebooks. Only two Python bytecode cache
files are omitted. `provenance.json` maps retained paths, sizes and SHA256
identities, lists the two exclusions and links each source/executed notebook.
The original P3/P4 manifests remain unchanged, including excluded cache entries;
the outer `SHA256SUMS.txt` covers every retained file plus the integration records.

Source notebooks remove outputs, execution counts and execution metadata.
P2's mutable `main` model lookup is changed to the observed immutable revision
for future replay; that exact cell substitution is declared and validated.
P3/P4 cell sources match their executed forms. P3 is a historical composite
with reused outcomes, not a fresh-source canonical repetition. P4 is final for
this supplementary protocol, not equivalent to M4's stricter repetition design.
No hidden patch cell is apparent; saved-source execution identity cannot be
proved from notebooks alone without a prior source freeze. Runner source,
logs, manifests and derived outputs do agree. Preliminary notebook narrative
is retained as history, not elevated above the machine data.

## Archive and checkout audit

- ALLaM ZIP: `kaggle-vllm-0.2.0-PR-29-AllaM.zip`, 2,024,325 bytes,
  SHA256 `f849a817476fa13b344fd21e233b349f27fbad29d1c5c2385411b0df8e03eac8`.
- Main ZIP: `kaggle-vllm-main.zip`, 7,042,485 bytes,
  SHA256 `baedb034a7fd18c05636480afa566ca7f833b6dbd2a22bea274f6641bd6887b9`.
- Local-project ZIP: `kaggle-vllm.zip`, 238,878,723 bytes,
  SHA256 `44ed9f8a27f225ae9ccda5d6ecc6207676b34c8d3021ddd80685e2b6e3f29665`.

The first two were in Downloads. The third was found beside the working
repository, not in Downloads. All 571 tracked main files match both repository
ZIPs and the active checkout byte-for-byte at main
`39f194eebf56c44e0f56bff54c1123a41725c049`; the active branch itself is the
merged PR26 head `eecf6b0fc248382bf14475dcf6970e5b1a6f3638`.
The local ZIP contains 10,560 files, including 9,580 `.local-evidence` entries,
`.git`, caches, distributions and unrelated notebooks. It was inspected as a
reference and never unpacked over the repository. Its tracked content has no
divergence; its local-only files were not imported.

The active checkout's untracked `ai-agents-nv-notebooks/` and Milestone-X
notebook were preserved in place. Integration uses a separate worktree.
PR29's deleted branch was recovered from its exact pull ref, preserving
`c5c3fab7a46b88440b8f64fe3ad0cf234ff4f01b` and its main ancestry. The old uploaded
notebook is the P2 execution, not the three-stage study.

Complete local inventories recorded every archive path, size, SHA256 and
extension; scans covered credential patterns, developer paths and large files.
No credential patterns were found. Existing developer paths in historical M4
and local-only material were left unchanged. ALLaM has no developer absolute
paths, weights, native wheels, HF snapshot files or credentials. Legitimate
Kaggle/runtime paths remain in immutable evidence; output paths for startup
failures intentionally need not exist. Archive identities are recorded here;
the ZIPs themselves remain owner-held local material and have no public URL.
All evidence required for this case-study analysis is retained here.

## Error and cleanup disposition

All three notebooks validate as nbformat v4 with zero Jupyter error outputs.
Searches of outputs and logs for exceptions, errors, OOM, killed processes,
assertions and non-finite values were reviewed against result classifications.
P3 TP1/2048 and TP1/4096, and P4 TP1/4096, contain expected KV-cache startup
tracebacks and are retained as RESOURCE_GATED; no performance is imputed.
FA2 rejection is logged at ERROR severity, followed by successful TRITON_ATTN
selection; it is not an accepted-run failure. Notebook dataframe displays use
NaN for unavailable fields; raw JSON uses null and CSV uses blanks. FP16 casting
and shutdown warnings remain visible. TP1 process-group-destruction warnings prevent a claim of perfectly
clean resource teardown. P2's final process/GPU listing is empty; P4 viable
servers report return code 0 and shutdown logs. Capacity, compatibility,
performance and language quality are separate claims.

## Reproduce and verify (CPU only)

From the repository root:

```bash
python scripts/verify_allam_evidence.py
python -m pytest -q tests/test_allam_evidence.py
(cd artifacts/kaggle-2026-10-01-allam-7b && sha256sum -c SHA256SUMS.txt)
```

Optional full historical analysis replay requires pandas and matplotlib:

```bash
python scripts/reproduce_allam_analysis.py --output /tmp/allam-analysis-replay
```

This executes only the explicitly selected, reviewed CPU analysis cells from
the hash-verified notebooks. It regenerates all 15 CSV tables, both findings
JSON files and eleven plots, and checks numeric/text agreement with retained
tables. No model download, server or GPU code runs. Plot input values and
recipes reproduce; historical plotting-library/font versions were not recorded,
so original PNG byte identity is not promised. PNG originals remain checksummed.

`python scripts/verify_allam_evidence.py --write-summary` regenerates the small
research-facing summary and results table. The validator independently checks
wave denominators, actual tokens, request TPOT and interpolated quantiles,
model/runtime pins, both TP ranks, complete matrices, original manifests,
runner-source linkage and output-free notebook transformations.

This changes research evidence/docs and CPU audit tooling only. Package 0.2.0,
PyPI releases, native wheels, historical checksums and M1–M4 authority remain
unchanged. No model material or paper PDF is redistributed. See the report's
ALLaM-specific license review; this is a technical classification, not legal advice.
