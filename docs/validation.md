# Validation record

The source evidence is the immutable local `Kaggle-Session-Files-3` collection.
The small Git-safe subset under `artifacts/kaggle-2026-08-23/` records
provenance, environment, identities, checksums, concise logs, and API responses.

## Passed Kaggle gates

| Gate | Evidence-backed result |
|---|---|
| Wheel SHA256 | PASS |
| Staged wheel install (`--no-deps`) | PASS |
| Native extension imports | PASS |
| Isolated dependency overlay | PASS |
| Existing PyTorch retained | PASS |
| Single Tesla T4 inference | PASS |
| Raw NCCL two-rank all-reduce | PASS |
| vLLM TP=2 inference | PASS |
| Qwen2.5-3B-Instruct FP16 TP=2 | PASS |
| TP-aware sharded-state save | PASS |
| New-engine sharded-state reload | PASS |
| OpenAI `/v1/models` | HTTP 200 |
| OpenAI `/v1/chat/completions` | HTTP 200 |

The generated text quality from the tiny OPT smoke model is not used as a
correctness claim; engine creation, distributed execution, and non-null
generation were the gates.

## Local tests

CPU tests mock PyTorch/vLLM boundaries. The real GPU profile test is marked
`gpu` and `kaggle` and skips outside the required runtime. Local tests must not
be presented as a substitute for the archived GPU run.

```bash
python3 -m compileall src tests
python3 -m pytest -q
```
