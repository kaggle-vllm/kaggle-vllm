# Canonical Phi-4 Mini M4 compatibility evidence

Status: **COMPATIBILITY_PASS**

This directory is the reviewed output of one fresh Kaggle T4 x2 execution of
`compat-phi4_mini`. The clean checkout used implementation commit
`42bf096c032e2c6be1e2fa3d573c7c86ac589ba2` and kaggle-vllm 0.2.0.

The downloaded ZIP and executed notebook remain outside Git. Their immutable
external identities are:

- ZIP SHA256:
  `7a20d7058cdcf7362704acbc5513db65eb865b864fd47b555f72465a32a0ebfe`
- ZIP size: 122549 bytes
- Executed notebook SHA256:
  `878c9b980787acec04ede93071771465f9897acd412006afc60277d92893e3b9`
- Executed notebook size: 40248 bytes
- Frozen source notebook SHA256:
  `885c5b6278614c66ac72f1e9ed8d05dbad84a19c8b99be129ebeb25e48a91dd9`

Both `sha256sum -c SHA256SUMS.txt` and the repository `verify-hashes` command
passed for all 16 evidence payloads. The executed notebook's four cell sources
match the frozen source notebook exactly; only execution counts, outputs, and
normal Kaggle metadata differ. It uses the single `M4_SHARD_ID` secret key and
contains no numbered variant.

TP1 and TP2 each completed one excluded warmup plus 20 measured requests with
zero failures. Every measured request reported exactly 128 input and 64 output
tokens. TP1 used GPU 0 while GPU 1 remained idle; TP2 used both GPUs. No OOM,
resource violation, semantic-gate failure, serving traceback, or NCCL failure
was observed.

`M4_PHI4_MINI_COMPATIBILITY_REVIEW.json` records the independent audit and the
non-averaging plausibility comparison with the prior rejected Phi candidate.
That prior candidate remains `REJECTED_NOTEBOOK_DRIFT`, is Git-excluded, and
is not promoted or combined with this evidence. This evidence supports
compatibility only; it does not establish a Phi crossover or complete the M4
principal matrix.
