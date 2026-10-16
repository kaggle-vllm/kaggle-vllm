# Native wheel verification

- L4: `vllm-0.30.0-cp312-cp312-linux_x86_64.whl`, 293855397 bytes, SHA256 `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`: local PASS. Hub LFS identity at `v0.3.0/l4x4/native/` matches. Source vLLM v0.30.0, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, SM89.
- T4: `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`, 383922260 bytes, SHA256 `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`: local PASS. Historical pinned Hub commit `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`, SM75.

No native wheel was rebuilt. Local Python 3.11 did not import either cp312 wheel. Historical GPU evidence applies to the exact previously tested artifacts only.
