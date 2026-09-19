# M4 execution reconciliation

Status: **AUTHORITATIVE TRACKED NO-RERUN LEDGER**.

Evidence takes precedence in this order: cryptographically verified preserved
evidence, verified local staging, tracked machine-readable status, PR prose,
then filenames or UI screenshots.

## Exact 60-shard state

- Canonical preserved: 38
- Failed resource gate: 4
- Failed other/review-required: 0
- Verified local staging pending promotion: 0
- Not executed: 18
- Total: 60 logical shards / 720 serving cells

Every canonical or reviewed terminal shard is excluded from normal continuation.
A resource-gated shard is settled negative evidence, not a zero-throughput result.

## No-rerun ledger

- `qwen25_3b-short-r00`
- `qwen25_3b-short-r01`
- `qwen25_3b-short-r02`
- `qwen25_3b-short-r03`
- `qwen25_3b-balanced-r00`
- `qwen25_3b-balanced-r01`
- `qwen25_3b-balanced-r02`
- `qwen25_3b-balanced-r03`
- `qwen25_3b-prefill_heavy-r00`
- `qwen25_3b-prefill_heavy-r01`
- `qwen25_3b-prefill_heavy-r02`
- `qwen25_3b-prefill_heavy-r03`
- `phi4_mini-short-r00`
- `phi4_mini-short-r01`
- `phi4_mini-short-r02`
- `phi4_mini-balanced-r00`
- `phi4_mini-balanced-r01`
- `phi4_mini-balanced-r02`
- `phi4_mini-prefill_heavy-r00`
- `phi4_mini-prefill_heavy-r01`
- `phi4_mini-prefill_heavy-r02`
- `llama32_3b-short-r00`
- `llama32_3b-short-r01`
- `llama32_3b-short-r02`
- `llama32_3b-balanced-r00`
- `llama32_3b-balanced-r01`
- `llama32_3b-balanced-r02`
- `llama32_3b-prefill_heavy-r00`
- `llama32_3b-prefill_heavy-r01`
- `llama32_3b-prefill_heavy-r02`
- `ministral3_3b_bf16-short-r00`
- `ministral3_3b_bf16-short-r01`
- `ministral3_3b_bf16-short-r02`
- `ministral3_3b_bf16-short-r03`
- `ministral3_3b_bf16-balanced-r00`
- `ministral3_3b_bf16-balanced-r01`
- `ministral3_3b_bf16-balanced-r02`
- `ministral3_3b_bf16-balanced-r03`
- `ministral3_3b_bf16-prefill_heavy-r00`
- `ministral3_3b_bf16-prefill_heavy-r01`
- `ministral3_3b_bf16-prefill_heavy-r02`
- `ministral3_3b_bf16-prefill_heavy-r03`

## Deterministic remaining batches

| Order | Batch ID | Rep | Model | Shards | Cells | Settled exclusions |
|---:|---|---:|---|---:|---:|---|
| 1 | `r03-phi` | 3 | `phi4_mini` | 3 | 36 | — |
| 2 | `r03-llama` | 3 | `llama32_3b` | 3 | 36 | — |
| 3 | `r04-qwen` | 4 | `qwen25_3b` | 3 | 36 | — |
| 4 | `r04-phi` | 4 | `phi4_mini` | 3 | 36 | — |
| 5 | `r04-llama` | 4 | `llama32_3b` | 3 | 36 | — |
| 6 | `r04-ministral` | 4 | `ministral3_3b_bf16` | 3 | 36 | — |

Next: `M4_BATCH_ID=r03-phi`.

The detailed shard IDs and order are machine-readable in
`M4_REMAINING_EXECUTION_PLAN.json`.
