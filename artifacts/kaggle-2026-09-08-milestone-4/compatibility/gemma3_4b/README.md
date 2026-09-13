# Canonical Gemma 3 4B M4 negative compatibility evidence

Status: **UNSUPPORTED_DTYPE_INTERSECTION_ON_SM75_FROZEN_STACK**

This directory preserves the hash-authenticated output of the final
`compat-gemma3_4b` Kaggle T4 x2 execution. The executed notebook's cell IDs,
types, order, and sources exactly match the frozen repository notebook. It ran
source commit `42bf096c032e2c6be1e2fa3d573c7c86ac589ba2`, kaggle-vllm 0.2.0, and
`google/gemma-3-4b-it` revision
`093f9f388b31de276ce2de164bdc2081324b9767` with the frozen `float16`
protocol.

The external identities are:

- evidence ZIP SHA256: `6404b8abe22fc06211409288e59ae1d844354c1b12282a8c9140161c30347c12`
- executed notebook SHA256: `f1800e9be84fa297880960159da906612e1ef79d46b2c34344d27f0bbe2feee8`
- runtime JSON SHA256: `8e747082b76d39852fa5838a3ce17f0ca8e392e7bac6ae4e0718700470e90ff5`
- frozen source notebook SHA256: `885c5b6278614c66ac72f1e9ed8d05dbad84a19c8b99be129ebeb25e48a91dd9`

The ZIP passed CRC, duplicate-name, traversal, absolute-path, symlink,
encryption, and member-set checks. Both `sha256sum -c SHA256SUMS.txt` and the
repository verifier passed all 16 payloads.

TP1 and TP2 independently resolved `Gemma3ForConditionalGeneration` and then
failed during model configuration before server readiness. Frozen vLLM rejects
Gemma3 `float16` for numerical stability and requests `bfloat16` or `float32`.
The T4 GPUs are SM75, while this frozen CUDA policy requires compute capability
8.0 or newer for BF16. FP32 was not run: it would materially change the memory
footprint and comparison protocol.

No measured requests were issued. Throughput and latency are therefore N/A,
not zero. No OOM, NCCL root-cause failure, authentication failure, dirty source,
or wheel mismatch occurred. The 15 historically planned Gemma principal shards
remain auditable but are `SKIPPED_BY_COMPATIBILITY_GATE`. Gemma 4 was not
substituted. This is a narrow result for the pinned model, revision, stack,
dtype, and SM75 hardware—not a universal Gemma 3 claim.
