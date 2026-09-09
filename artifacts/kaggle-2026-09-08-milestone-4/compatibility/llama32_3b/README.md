# Canonical Llama 3.2 3B M4 compatibility evidence

Status: **COMPATIBILITY_PASS**

This directory is the independently reviewed output of one fresh Kaggle T4 x2
execution of `compat-llama32_3b`. The clean checkout used implementation commit
`42bf096c032e2c6be1e2fa3d573c7c86ac589ba2`, kaggle-vllm 0.2.0, and model and
tokenizer revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`.

The downloaded ZIP, runtime manifest, and executed notebook remain outside
Git. Their immutable external identities are:

- ZIP SHA256:
  `37288c24065ddd2383c5e8dfebf08b61ca589bb2b8a37e9ae253ef9e6a1f9db4`
- ZIP size: 105270 bytes
- Executed notebook SHA256:
  `cf12cd6c829896f50ceaa5dcd71ea8c0fb9465ceacaa701069df5167e4157de5`
- Executed notebook size: 40273 bytes
- Runtime JSON SHA256:
  `f6f15d4998e56c9acd7390ef222d8dc6bcd9bb8148149d1bf507f66c106a0577`
- Frozen source notebook SHA256:
  `885c5b6278614c66ac72f1e9ed8d05dbad84a19c8b99be129ebeb25e48a91dd9`

Both `sha256sum -c SHA256SUMS.txt` and the repository `verify-hashes` command
passed for all 16 evidence payloads. All four cell sources, cell types, IDs,
and order match the frozen repository notebook exactly. The executed notebook
uses only the unnumbered `M4_SHARD_ID`, reports runner return code 0, uses the
ordinary `/kaggle/working/m4-evidence` root, and verifies the evidence hashes.

TP1 and TP2 each completed one excluded warmup plus 20 measured requests with
zero failures. Every measured request reported exactly 128 input and 64 output
tokens. TP1 used GPU 0 while GPU 1 remained idle; TP2 initialized two NCCL
ranks and used both GPUs. No OOM, resource violation, semantic-gate failure,
authentication failure, serving traceback, or NCCL failure was observed.

The logs resolve `LlamaForCausalLM` at the exact pinned revision with
`trust_remote_code=False` and no quantization. FlashAttention-2 rejection is
logged at error severity because T4 is SM75; vLLM then selects `TRITON_ATTN`
and completes both cells. TP2 also reports SymmMem unavailability on SM75
before successful symmetric NCCL 2.27.5 initialization. These are retained as
non-blocking scientific limitations of the measured eager Triton path.

The earlier manually edited access/retry run remains Git-excluded
`DEBUG_COMPATIBILITY_PASS` evidence and is not averaged with this run. Its TP2
metrics are close to the clean run. Its slower TP1 coincided with substantially
lower sampled clocks and higher temperature, making session thermal/clock
state a plausible explanation rather than an identity or semantic mismatch.

`M4_LLAMA32_3B_COMPATIBILITY_REVIEW.json` records the complete audit. This
evidence establishes only the short-workload concurrency-1 compatibility gate;
it does not establish a Llama crossover, complete the principal matrix, finish
M4, or make the work paper-ready.
