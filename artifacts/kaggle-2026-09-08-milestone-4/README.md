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

The prior Phi candidate remains separately classified `REJECTED_NOTEBOOK_DRIFT`
and is not canonical evidence. Not yet executed: Llama 3.2 3B, Ministral 3 3B
BF16, and Gemma 3 4B compatibility shards. No M4 principal shard is accepted
here.
