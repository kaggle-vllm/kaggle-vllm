# M4 principal execution queue

Compatibility is closed. Use the unchanged frozen notebook in one fresh
Kaggle T4 x2 session per row and change only the `M4_SHARD_ID` secret value.
For every active row, also download the executed copy of
`kaggle_vllm_m4_execute_shard.ipynb` and the separately generated
`/kaggle/working/kaggle-vllm-runtime/runtime.json`; both are mandatory
inputs to the local provenance audit.

Progress: 3 / 60 active shards preserved.
Review required: 1 resource-gated shard.

Next: `M4_SHARD_ID=phi4_mini-short-r00`

| Active | M4_SHARD_ID | Model | Workload | Rep | Status | Artifact |
|---:|---|---|---|---:|---|---|
| 1 | `qwen25_3b-short-r00` | `qwen25_3b` | short | 0 | PRINCIPAL_SHARD_PRESERVED | `qwen25_3b-short-r00-principal.zip` |
| 2 | `qwen25_3b-short-r01` | `qwen25_3b` | short | 1 | PRINCIPAL_SHARD_PRESERVED | `qwen25_3b-short-r01-principal.zip` |
| 3 | `qwen25_3b-short-r02` | `qwen25_3b` | short | 2 | QUEUED | `qwen25_3b-short-r02-principal.zip` |
| 4 | `qwen25_3b-short-r03` | `qwen25_3b` | short | 3 | QUEUED | `qwen25_3b-short-r03-principal.zip` |
| 5 | `qwen25_3b-short-r04` | `qwen25_3b` | short | 4 | QUEUED | `qwen25_3b-short-r04-principal.zip` |
| 6 | `qwen25_3b-balanced-r00` | `qwen25_3b` | balanced | 0 | PRINCIPAL_SHARD_PRESERVED | `qwen25_3b-balanced-r00-principal.zip` |
| 7 | `qwen25_3b-balanced-r01` | `qwen25_3b` | balanced | 1 | QUEUED | `qwen25_3b-balanced-r01-principal.zip` |
| 8 | `qwen25_3b-balanced-r02` | `qwen25_3b` | balanced | 2 | QUEUED | `qwen25_3b-balanced-r02-principal.zip` |
| 9 | `qwen25_3b-balanced-r03` | `qwen25_3b` | balanced | 3 | QUEUED | `qwen25_3b-balanced-r03-principal.zip` |
| 10 | `qwen25_3b-balanced-r04` | `qwen25_3b` | balanced | 4 | QUEUED | `qwen25_3b-balanced-r04-principal.zip` |
| 11 | `qwen25_3b-prefill_heavy-r00` | `qwen25_3b` | prefill_heavy | 0 | FAILED_RESOURCE_GATE | `qwen25_3b-prefill_heavy-r00-principal.zip` |
| 12 | `qwen25_3b-prefill_heavy-r01` | `qwen25_3b` | prefill_heavy | 1 | QUEUED | `qwen25_3b-prefill_heavy-r01-principal.zip` |
| 13 | `qwen25_3b-prefill_heavy-r02` | `qwen25_3b` | prefill_heavy | 2 | QUEUED | `qwen25_3b-prefill_heavy-r02-principal.zip` |
| 14 | `qwen25_3b-prefill_heavy-r03` | `qwen25_3b` | prefill_heavy | 3 | QUEUED | `qwen25_3b-prefill_heavy-r03-principal.zip` |
| 15 | `qwen25_3b-prefill_heavy-r04` | `qwen25_3b` | prefill_heavy | 4 | QUEUED | `qwen25_3b-prefill_heavy-r04-principal.zip` |
| 16 | `phi4_mini-short-r00` | `phi4_mini` | short | 0 | QUEUED | `phi4_mini-short-r00-principal.zip` |
| 17 | `phi4_mini-short-r01` | `phi4_mini` | short | 1 | QUEUED | `phi4_mini-short-r01-principal.zip` |
| 18 | `phi4_mini-short-r02` | `phi4_mini` | short | 2 | QUEUED | `phi4_mini-short-r02-principal.zip` |
| 19 | `phi4_mini-short-r03` | `phi4_mini` | short | 3 | QUEUED | `phi4_mini-short-r03-principal.zip` |
| 20 | `phi4_mini-short-r04` | `phi4_mini` | short | 4 | QUEUED | `phi4_mini-short-r04-principal.zip` |
| 21 | `phi4_mini-balanced-r00` | `phi4_mini` | balanced | 0 | QUEUED | `phi4_mini-balanced-r00-principal.zip` |
| 22 | `phi4_mini-balanced-r01` | `phi4_mini` | balanced | 1 | QUEUED | `phi4_mini-balanced-r01-principal.zip` |
| 23 | `phi4_mini-balanced-r02` | `phi4_mini` | balanced | 2 | QUEUED | `phi4_mini-balanced-r02-principal.zip` |
| 24 | `phi4_mini-balanced-r03` | `phi4_mini` | balanced | 3 | QUEUED | `phi4_mini-balanced-r03-principal.zip` |
| 25 | `phi4_mini-balanced-r04` | `phi4_mini` | balanced | 4 | QUEUED | `phi4_mini-balanced-r04-principal.zip` |
| 26 | `phi4_mini-prefill_heavy-r00` | `phi4_mini` | prefill_heavy | 0 | QUEUED | `phi4_mini-prefill_heavy-r00-principal.zip` |
| 27 | `phi4_mini-prefill_heavy-r01` | `phi4_mini` | prefill_heavy | 1 | QUEUED | `phi4_mini-prefill_heavy-r01-principal.zip` |
| 28 | `phi4_mini-prefill_heavy-r02` | `phi4_mini` | prefill_heavy | 2 | QUEUED | `phi4_mini-prefill_heavy-r02-principal.zip` |
| 29 | `phi4_mini-prefill_heavy-r03` | `phi4_mini` | prefill_heavy | 3 | QUEUED | `phi4_mini-prefill_heavy-r03-principal.zip` |
| 30 | `phi4_mini-prefill_heavy-r04` | `phi4_mini` | prefill_heavy | 4 | QUEUED | `phi4_mini-prefill_heavy-r04-principal.zip` |
| 31 | `llama32_3b-short-r00` | `llama32_3b` | short | 0 | QUEUED | `llama32_3b-short-r00-principal.zip` |
| 32 | `llama32_3b-short-r01` | `llama32_3b` | short | 1 | QUEUED | `llama32_3b-short-r01-principal.zip` |
| 33 | `llama32_3b-short-r02` | `llama32_3b` | short | 2 | QUEUED | `llama32_3b-short-r02-principal.zip` |
| 34 | `llama32_3b-short-r03` | `llama32_3b` | short | 3 | QUEUED | `llama32_3b-short-r03-principal.zip` |
| 35 | `llama32_3b-short-r04` | `llama32_3b` | short | 4 | QUEUED | `llama32_3b-short-r04-principal.zip` |
| 36 | `llama32_3b-balanced-r00` | `llama32_3b` | balanced | 0 | QUEUED | `llama32_3b-balanced-r00-principal.zip` |
| 37 | `llama32_3b-balanced-r01` | `llama32_3b` | balanced | 1 | QUEUED | `llama32_3b-balanced-r01-principal.zip` |
| 38 | `llama32_3b-balanced-r02` | `llama32_3b` | balanced | 2 | QUEUED | `llama32_3b-balanced-r02-principal.zip` |
| 39 | `llama32_3b-balanced-r03` | `llama32_3b` | balanced | 3 | QUEUED | `llama32_3b-balanced-r03-principal.zip` |
| 40 | `llama32_3b-balanced-r04` | `llama32_3b` | balanced | 4 | QUEUED | `llama32_3b-balanced-r04-principal.zip` |
| 41 | `llama32_3b-prefill_heavy-r00` | `llama32_3b` | prefill_heavy | 0 | QUEUED | `llama32_3b-prefill_heavy-r00-principal.zip` |
| 42 | `llama32_3b-prefill_heavy-r01` | `llama32_3b` | prefill_heavy | 1 | QUEUED | `llama32_3b-prefill_heavy-r01-principal.zip` |
| 43 | `llama32_3b-prefill_heavy-r02` | `llama32_3b` | prefill_heavy | 2 | QUEUED | `llama32_3b-prefill_heavy-r02-principal.zip` |
| 44 | `llama32_3b-prefill_heavy-r03` | `llama32_3b` | prefill_heavy | 3 | QUEUED | `llama32_3b-prefill_heavy-r03-principal.zip` |
| 45 | `llama32_3b-prefill_heavy-r04` | `llama32_3b` | prefill_heavy | 4 | QUEUED | `llama32_3b-prefill_heavy-r04-principal.zip` |
| 46 | `ministral3_3b_bf16-short-r00` | `ministral3_3b_bf16` | short | 0 | QUEUED | `ministral3_3b_bf16-short-r00-principal.zip` |
| 47 | `ministral3_3b_bf16-short-r01` | `ministral3_3b_bf16` | short | 1 | QUEUED | `ministral3_3b_bf16-short-r01-principal.zip` |
| 48 | `ministral3_3b_bf16-short-r02` | `ministral3_3b_bf16` | short | 2 | QUEUED | `ministral3_3b_bf16-short-r02-principal.zip` |
| 49 | `ministral3_3b_bf16-short-r03` | `ministral3_3b_bf16` | short | 3 | QUEUED | `ministral3_3b_bf16-short-r03-principal.zip` |
| 50 | `ministral3_3b_bf16-short-r04` | `ministral3_3b_bf16` | short | 4 | QUEUED | `ministral3_3b_bf16-short-r04-principal.zip` |
| 51 | `ministral3_3b_bf16-balanced-r00` | `ministral3_3b_bf16` | balanced | 0 | QUEUED | `ministral3_3b_bf16-balanced-r00-principal.zip` |
| 52 | `ministral3_3b_bf16-balanced-r01` | `ministral3_3b_bf16` | balanced | 1 | QUEUED | `ministral3_3b_bf16-balanced-r01-principal.zip` |
| 53 | `ministral3_3b_bf16-balanced-r02` | `ministral3_3b_bf16` | balanced | 2 | QUEUED | `ministral3_3b_bf16-balanced-r02-principal.zip` |
| 54 | `ministral3_3b_bf16-balanced-r03` | `ministral3_3b_bf16` | balanced | 3 | QUEUED | `ministral3_3b_bf16-balanced-r03-principal.zip` |
| 55 | `ministral3_3b_bf16-balanced-r04` | `ministral3_3b_bf16` | balanced | 4 | QUEUED | `ministral3_3b_bf16-balanced-r04-principal.zip` |
| 56 | `ministral3_3b_bf16-prefill_heavy-r00` | `ministral3_3b_bf16` | prefill_heavy | 0 | QUEUED | `ministral3_3b_bf16-prefill_heavy-r00-principal.zip` |
| 57 | `ministral3_3b_bf16-prefill_heavy-r01` | `ministral3_3b_bf16` | prefill_heavy | 1 | QUEUED | `ministral3_3b_bf16-prefill_heavy-r01-principal.zip` |
| 58 | `ministral3_3b_bf16-prefill_heavy-r02` | `ministral3_3b_bf16` | prefill_heavy | 2 | QUEUED | `ministral3_3b_bf16-prefill_heavy-r02-principal.zip` |
| 59 | `ministral3_3b_bf16-prefill_heavy-r03` | `ministral3_3b_bf16` | prefill_heavy | 3 | QUEUED | `ministral3_3b_bf16-prefill_heavy-r03-principal.zip` |
| 60 | `ministral3_3b_bf16-prefill_heavy-r04` | `ministral3_3b_bf16` | prefill_heavy | 4 | QUEUED | `ministral3_3b_bf16-prefill_heavy-r04-principal.zip` |
| — | `gemma3_4b-short-r00` | `gemma3_4b` | short | 0 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-short-r00-principal.zip` |
| — | `gemma3_4b-short-r01` | `gemma3_4b` | short | 1 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-short-r01-principal.zip` |
| — | `gemma3_4b-short-r02` | `gemma3_4b` | short | 2 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-short-r02-principal.zip` |
| — | `gemma3_4b-short-r03` | `gemma3_4b` | short | 3 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-short-r03-principal.zip` |
| — | `gemma3_4b-short-r04` | `gemma3_4b` | short | 4 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-short-r04-principal.zip` |
| — | `gemma3_4b-balanced-r00` | `gemma3_4b` | balanced | 0 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-balanced-r00-principal.zip` |
| — | `gemma3_4b-balanced-r01` | `gemma3_4b` | balanced | 1 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-balanced-r01-principal.zip` |
| — | `gemma3_4b-balanced-r02` | `gemma3_4b` | balanced | 2 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-balanced-r02-principal.zip` |
| — | `gemma3_4b-balanced-r03` | `gemma3_4b` | balanced | 3 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-balanced-r03-principal.zip` |
| — | `gemma3_4b-balanced-r04` | `gemma3_4b` | balanced | 4 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-balanced-r04-principal.zip` |
| — | `gemma3_4b-prefill_heavy-r00` | `gemma3_4b` | prefill_heavy | 0 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-prefill_heavy-r00-principal.zip` |
| — | `gemma3_4b-prefill_heavy-r01` | `gemma3_4b` | prefill_heavy | 1 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-prefill_heavy-r01-principal.zip` |
| — | `gemma3_4b-prefill_heavy-r02` | `gemma3_4b` | prefill_heavy | 2 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-prefill_heavy-r02-principal.zip` |
| — | `gemma3_4b-prefill_heavy-r03` | `gemma3_4b` | prefill_heavy | 3 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-prefill_heavy-r03-principal.zip` |
| — | `gemma3_4b-prefill_heavy-r04` | `gemma3_4b` | prefill_heavy | 4 | SKIPPED_BY_COMPATIBILITY_GATE | `gemma3_4b-prefill_heavy-r04-principal.zip` |

Every active shard contains concurrency 1, 4, 8, 16, 32, and 64 for
TP1 and TP2 (12 serving cells). The 15 Gemma rows preserve frozen
historical intent and are not executable. Their performance is N/A.
