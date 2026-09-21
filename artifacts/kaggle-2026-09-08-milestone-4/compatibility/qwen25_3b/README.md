# Canonical Qwen2.5-3B M4 compatibility evidence

Status: **COMPATIBILITY_PASS**

This directory is the reviewed output of one fresh Kaggle T4 x2 execution of
`compat-qwen25_3b`. The clean checkout used implementation commit
`42bf096c032e2c6be1e2fa3d573c7c86ac589ba2` and kaggle-vllm 0.2.0.

The downloaded ZIP and executed notebook remain outside Git. Their immutable
external identities are:

- ZIP SHA256: `cd3c45dddf19830649b03a931cee0247d6f8b4ebbc953c433e3ade563b271ecb`
- ZIP size: 119276 bytes
- Executed notebook SHA256:
  `d2106fcbd0df37cf34789499b90d71644c93b486d43265cc67c921b5ba91a94f`
- Executed notebook size: 40354 bytes
- Frozen source notebook SHA256:
  `885c5b6278614c66ac72f1e9ed8d05dbad84a19c8b99be129ebeb25e48a91dd9`

Both `sha256sum -c SHA256SUMS.txt` and the repository `verify-hashes` command
passed for all 16 evidence payloads. The executed notebook's four cell sources
match the frozen source notebook exactly; only execution counts, outputs, and
normal Kaggle metadata differ.

TP1 and TP2 each completed one excluded warmup plus 20 measured requests with
zero failures. Every measured request reported exactly 128 input and 64 output
tokens. TP1 used GPU 0 while GPU 1 remained idle; TP2 used both GPUs. No OOM,
resource violation, semantic-gate failure, serving traceback, or NCCL failure
was observed.

`M4_QWEN_COMPATIBILITY_REVIEW.json` records the independent audit and the
non-averaging plausibility comparison with the earlier
`DEBUG_COMPATIBILITY_PASS` run. This evidence supports compatibility only; it
does not establish an M4 crossover or complete the principal matrix.
