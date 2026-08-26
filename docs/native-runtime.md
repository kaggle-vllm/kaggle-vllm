# Native runtime

The native artifact is a separate upstream vLLM CUDA wheel, never a normal SDK
dependency:

```text
vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl
SHA256 5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c
HF revision f6b4f10de54924ed6fe9e28cceab84eca7276ab6
```

The wheel tag is `cp312-cp312-linux_x86_64`, `Root-Is-Purelib: false`, and its
metadata reports version `0.18.2.dev0+ga26e8dc7f.d20260822.cu128`. Source
provenance remains upstream tag v0.18.1 at
`a26e8dc7ff2111a005144d775ecf9cebf56c45b2`.

## Structural inspection

Five compiled ELF64 x86-64 shared objects were observed:

- `vllm/_C.abi3.so`
- `vllm/_moe_C.abi3.so`
- `vllm/cumem_allocator.abi3.so`
- `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`
- `vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so`

`readelf` records dynamic links to the expected Torch libraries, CUDA runtime
12 and/or live `libcuda.so.1`, plus ordinary C/C++ runtime libraries. No
RPATH/RUNPATH entries were observed. `cuobjdump` found `sm_75` ELF images in
`_C` and `_moe_C`, supporting the intended T4 target. The bundled FlashAttention
objects do not imply FlashAttention 2 is usable on SM75; runtime evidence shows
vLLM selecting `TRITON_ATTN`.

Use `scripts/inspect_wheel.sh` or standard ZIP/ELF tools against a copy or
read-only path. Never patch, rename or recompress the authoritative wheel.

## Staging and activation

Bootstrap downloads into a checksum-verified cache, stages the wheel with
`pip --target --no-deps`, stages the locked overlay separately and atomically
writes a runtime manifest. Activation prepends only those owned paths and the
existing Torch library directory. It does not edit shell startup files.

The wheel's full dependency metadata is broader than the curated diagnostic
subset. The overlay and dependency baseline capture the packages implicated in
the validated workflow; future wheel changes require regenerating and reviewing
both rather than assuming forward compatibility.
