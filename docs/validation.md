# Validation record

The source evidence is the immutable local `Kaggle-Session-Files-3` collection.
The small Git-safe subset under `artifacts/kaggle-2026-08-23/` records
provenance, environment, identities, checksums, concise logs, and API responses.
The post-publication 2026-08-25 run is preserved as an
[executed acceptance notebook](../kaggle-notebooks/kaggle_vllm_0_1_1_dual_t4_acceptance.ipynb)
with a [concise evidence report](kaggle-v0.1.1-acceptance.md).

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

The fresh v0.1.1 run additionally verified strict bootstrap from the renamed
Hugging Face repository, staged native imports, OPT TP=2 generation, and Qwen
TP=2 `sharded_state` generation. Generated text quality is not used as a
correctness claim; engine creation, distributed execution, and non-null
generation were the gates.

## Local tests

CPU tests mock PyTorch/vLLM boundaries. The real GPU profile test is marked
`gpu` and `kaggle` and skips outside the required runtime. Local tests must not
be presented as a substitute for the executed Kaggle GPU evidence.

```bash
python3 -m compileall src tests
python3 -m pytest -q
```
