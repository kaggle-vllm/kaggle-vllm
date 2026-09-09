# Milestone 4 evidence

Status: **IN_PROGRESS**

This directory contains accepted, checksum-verified Milestone 4 evidence. It
does not imply that the compatibility program, principal matrix, Milestone 4,
or the paper is complete.

Accepted compatibility shards:

- `compatibility/qwen25_3b/`: `COMPATIBILITY_PASS` for the short-workload,
  concurrency-1 TP1/TP2 gate on a fresh Kaggle T4 x2 session.
- `compatibility/phi4_mini/`: `COMPATIBILITY_PASS` for the short-workload,
  concurrency-1 TP1/TP2 gate from the independently audited clean rerun.
- `compatibility/llama32_3b/`: `COMPATIBILITY_PASS` for the short-workload,
  concurrency-1 TP1/TP2 gate from the clean canonical Llama 3.2 run.
- `compatibility/ministral3_3b_bf16/`: `COMPATIBILITY_PASS` for the text-only
  short-workload, concurrency-1 TP1/TP2 gate on the multimodal-capable model.

The prior Phi candidate remains separately classified `REJECTED_NOTEBOOK_DRIFT`
and is not canonical evidence. The Llama access-pending and token-authorization
failures remain access records rather than compatibility failures. Its later
manually edited retry remains excluded `DEBUG_COMPATIBILITY_PASS` evidence and
was not averaged with the accepted clean run. Gemma remains `NOT_EXECUTED`, and
no M4 principal shard is accepted here.
