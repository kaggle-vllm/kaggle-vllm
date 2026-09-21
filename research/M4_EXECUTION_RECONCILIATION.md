# M4 execution reconciliation

Status: **AUTHORITATIVE TRACKED NO-RERUN LEDGER**.

Evidence takes precedence in this order: cryptographically verified preserved
evidence, verified local staging, tracked machine-readable status, PR prose,
then filenames or UI screenshots.

## Exact 60-shard state

- Canonical preserved: 55
- Failed resource gate: 5
- Failed other/review-required: 0
- Verified local staging pending promotion: 0
- Not executed: 0
- Total: 60 logical shards / 720 serving cells

Every canonical or reviewed terminal shard is excluded from normal continuation.
A resource-gated shard is settled negative evidence, not a zero-throughput result.

## No-rerun ledger

- `qwen25_3b-short-r00`
- `qwen25_3b-short-r01`
- `qwen25_3b-short-r02`
- `qwen25_3b-short-r03`
- `qwen25_3b-short-r04`
- `qwen25_3b-balanced-r00`
- `qwen25_3b-balanced-r01`
- `qwen25_3b-balanced-r02`
- `qwen25_3b-balanced-r03`
- `qwen25_3b-balanced-r04`
- `qwen25_3b-prefill_heavy-r00`
- `qwen25_3b-prefill_heavy-r01`
- `qwen25_3b-prefill_heavy-r02`
- `qwen25_3b-prefill_heavy-r03`
- `qwen25_3b-prefill_heavy-r04`
- `phi4_mini-short-r00`
- `phi4_mini-short-r01`
- `phi4_mini-short-r02`
- `phi4_mini-short-r03`
- `phi4_mini-short-r04`
- `phi4_mini-balanced-r00`
- `phi4_mini-balanced-r01`
- `phi4_mini-balanced-r02`
- `phi4_mini-balanced-r03`
- `phi4_mini-balanced-r04`
- `phi4_mini-prefill_heavy-r00`
- `phi4_mini-prefill_heavy-r01`
- `phi4_mini-prefill_heavy-r02`
- `phi4_mini-prefill_heavy-r03`
- `phi4_mini-prefill_heavy-r04`
- `llama32_3b-short-r00`
- `llama32_3b-short-r01`
- `llama32_3b-short-r02`
- `llama32_3b-short-r03`
- `llama32_3b-short-r04`
- `llama32_3b-balanced-r00`
- `llama32_3b-balanced-r01`
- `llama32_3b-balanced-r02`
- `llama32_3b-balanced-r03`
- `llama32_3b-balanced-r04`
- `llama32_3b-prefill_heavy-r00`
- `llama32_3b-prefill_heavy-r01`
- `llama32_3b-prefill_heavy-r02`
- `llama32_3b-prefill_heavy-r03`
- `llama32_3b-prefill_heavy-r04`
- `ministral3_3b_bf16-short-r00`
- `ministral3_3b_bf16-short-r01`
- `ministral3_3b_bf16-short-r02`
- `ministral3_3b_bf16-short-r03`
- `ministral3_3b_bf16-short-r04`
- `ministral3_3b_bf16-balanced-r00`
- `ministral3_3b_bf16-balanced-r01`
- `ministral3_3b_bf16-balanced-r02`
- `ministral3_3b_bf16-balanced-r03`
- `ministral3_3b_bf16-balanced-r04`
- `ministral3_3b_bf16-prefill_heavy-r00`
- `ministral3_3b_bf16-prefill_heavy-r01`
- `ministral3_3b_bf16-prefill_heavy-r02`
- `ministral3_3b_bf16-prefill_heavy-r03`
- `ministral3_3b_bf16-prefill_heavy-r04`

## Deterministic remaining batches

| Order | Batch ID | Rep | Model | Shards | Cells | Settled exclusions |
|---:|---|---:|---|---:|---:|---|

Next: no batch remains; no further GPU execution is authorized.

The detailed shard IDs and order are machine-readable in
`M4_REMAINING_EXECUTION_PLAN.json`.
