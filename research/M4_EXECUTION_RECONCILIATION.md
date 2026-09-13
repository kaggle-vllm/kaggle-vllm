# M4 execution reconciliation

Status: **AUTHORITATIVE TRACKED NO-RERUN LEDGER**.

Evidence takes precedence in this order: cryptographically verified preserved
evidence, verified local staging, tracked machine-readable status, PR prose,
then filenames or UI screenshots.

## Exact 60-shard state

- Canonical preserved: 15
- Failed resource gate: 1
- Failed other/review-required: 0
- Verified local staging pending promotion: 0
- Not executed: 44
- Total: 60 logical shards / 720 serving cells

Every canonical or reviewed terminal shard is excluded from normal continuation.
A resource-gated shard is settled negative evidence, not a zero-throughput result.

## No-rerun ledger

- `qwen25_3b-short-r00`
- `qwen25_3b-short-r01`
- `qwen25_3b-balanced-r00`
- `qwen25_3b-prefill_heavy-r00`
- `phi4_mini-short-r00`
- `phi4_mini-short-r01`
- `phi4_mini-balanced-r00`
- `phi4_mini-balanced-r01`
- `phi4_mini-prefill_heavy-r00`
- `phi4_mini-prefill_heavy-r01`
- `llama32_3b-short-r00`
- `llama32_3b-balanced-r00`
- `llama32_3b-prefill_heavy-r00`
- `ministral3_3b_bf16-short-r00`
- `ministral3_3b_bf16-balanced-r00`
- `ministral3_3b_bf16-prefill_heavy-r00`

## Deterministic remaining batches

| Order | Batch ID | Rep | Model | Shards | Cells | Settled exclusions |
|---:|---|---:|---|---:|---:|---|
| 1 | `fill-r01-llama` | 1 | `llama32_3b` | 3 | 36 | — |
| 2 | `fill-r01-ministral` | 1 | `ministral3_3b_bf16` | 3 | 36 | — |
| 3 | `fill-r01-qwen` | 1 | `qwen25_3b` | 2 | 24 | `qwen25_3b-short-r01` |
| 4 | `r02-llama` | 2 | `llama32_3b` | 3 | 36 | — |
| 5 | `r02-ministral` | 2 | `ministral3_3b_bf16` | 3 | 36 | — |
| 6 | `r02-qwen` | 2 | `qwen25_3b` | 3 | 36 | — |
| 7 | `r02-phi` | 2 | `phi4_mini` | 3 | 36 | — |
| 8 | `r03-ministral` | 3 | `ministral3_3b_bf16` | 3 | 36 | — |
| 9 | `r03-qwen` | 3 | `qwen25_3b` | 3 | 36 | — |
| 10 | `r03-phi` | 3 | `phi4_mini` | 3 | 36 | — |
| 11 | `r03-llama` | 3 | `llama32_3b` | 3 | 36 | — |
| 12 | `r04-qwen` | 4 | `qwen25_3b` | 3 | 36 | — |
| 13 | `r04-phi` | 4 | `phi4_mini` | 3 | 36 | — |
| 14 | `r04-llama` | 4 | `llama32_3b` | 3 | 36 | — |
| 15 | `r04-ministral` | 4 | `ministral3_3b_bf16` | 3 | 36 | — |

Next: `M4_BATCH_ID=fill-r01-llama`.

The detailed shard IDs and order are machine-readable in
`M4_REMAINING_EXECUTION_PLAN.json`.
