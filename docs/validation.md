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

The 2026-08-30 `0.2.0.dev0` development-candidate acceptance additionally
verified the dependency-aware strict doctor, strict immutable delivery,
unchanged system PyTorch, native imports, raw NCCL, OPT TP=1 and TP=2, local
OpenAI-compatible HTTP 200 responses, and clean worker termination. Its source
identity is `7327b0b0c811a92a9c49421a4d302c18e251ab61`. The separate controlled
benchmark used `6d10912ad73e81f5a62fcec299c87ed5b2631b4f`; for OPT-125M it recorded
TP=1 faster than TP=2. Final acceptance of the package published as 0.2.0 is
still required.

## Local tests

CPU tests mock PyTorch/vLLM boundaries. The real GPU profile test is marked
`gpu` and `kaggle` and skips outside the required runtime. Local tests must not
be presented as a substitute for the executed Kaggle GPU evidence.

```bash
python3 -m compileall src tests
python3 -m pytest -q
```
