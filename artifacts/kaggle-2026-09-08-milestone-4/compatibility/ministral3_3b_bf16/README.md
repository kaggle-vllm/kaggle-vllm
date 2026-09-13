# Canonical Ministral 3 3B BF16 M4 compatibility evidence

Status: **COMPATIBILITY_PASS**

This directory is the independently reviewed output of one fresh Kaggle T4 x2
execution of `compat-ministral3_3b_bf16`. The clean checkout used implementation
commit `42bf096c032e2c6be1e2fa3d573c7c86ac589ba2`, kaggle-vllm 0.2.0, and model
revision `b6d637bef2393152b3da2b2fde72eecdee30557e`.

The downloaded ZIP and executed notebook remain outside Git. Their immutable
external identities are:

- ZIP SHA256:
  `516e37c160ca98a49795870aeeeeb8a381cd8780924f4d9d8836305ad68c77cc`
- ZIP size: 111387 bytes
- Executed notebook SHA256:
  `5f988e506a35a21f59c1502ae77eb5c6df9118549d4cd95ce7f0ea8a397c3a1d`
- Executed notebook size: 40791 bytes
- Frozen source notebook SHA256:
  `885c5b6278614c66ac72f1e9ed8d05dbad84a19c8b99be129ebeb25e48a91dd9`

Both `sha256sum -c SHA256SUMS.txt` and the repository `verify-hashes` command
passed for all 16 evidence payloads. The two executable cell sources match the
frozen repository notebook exactly. The executed notebook uses only the
unnumbered `M4_SHARD_ID`, reports runner return code 0, verifies the evidence
hashes, and prints the independently reproduced ZIP digest.

TP1 and TP2 each completed one excluded warmup plus 20 measured requests with
zero failures. Every measured request reported exactly 128 input and 64 output
tokens. TP1 used GPU 0 while GPU 1 remained idle; TP2 used both GPUs. No OOM,
resource violation, semantic-gate failure, serving traceback, or NCCL failure
was observed.

The model's top-level pinned configuration declares
`Mistral3ForConditionalGeneration`; vLLM resolves that multimodal architecture
to its native `PixtralForConditionalGeneration` implementation and performs an
encoder-cache warmup. The measured workload nevertheless used plain text on
the `/v1/completions` generation path with no image payload. Transformers 4.57.6
also warns about the checkpoint's legacy Mistral pre-tokenizer regex. An
independent CPU-only check at the pinned revision showed that applying
`fix_mistral_regex=True` changes none of the 64 retained prompt token sequences,
token-ID hashes, or 128-token counts. These behaviors bound interpretation to
this exact text-only workload and runtime; they do not block its compatibility
result.

`M4_MINISTRAL3_3B_BF16_COMPATIBILITY_REVIEW.json` records the complete audit.
This evidence establishes compatibility only. It does not establish a
Ministral crossover, complete the M4 principal matrix, or make the work
paper-ready.
