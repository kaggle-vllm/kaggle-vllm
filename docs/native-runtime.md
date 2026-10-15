# P31 native runtime and runtime-v3

Canonical final-v2 tar: `e82085d0c6f180d2f4f8eb8f52332a92051273715e0923082fc65fc8857ddad5` (291,899,459 bytes). The original checkpoint tar `2429f6c04ebb64e6119e219cfe99e03c665c8228ec23994f2ab4f5f179e14192` remains provenance only.

Native wheel: `vllm-0.30.0-cp312-cp312-linux_x86_64.whl`, 293,855,397 bytes, SHA256 `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`. Its ZIP contains `vllm/_C_stable_libtorch.abi3.so` and `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`. P31 import and SM89 build-log audits passed. The tested Gemma 4 architecture chose `TRITON_ATTN`; FA2 end-to-end model selection is a future optional acceptance test.

P31 build contract: `CMAKE_BUILD_TYPE=Release`, `CMAKE_CUDA_ARCHITECTURES=89`, `TORCH_CUDA_ARCH_LIST=8.9`, `FETCHCONTENT_FULLY_DISCONNECTED=ON`. The P31 source build used the output of `nproc --all` (48 in that session); normal v0.3.0 runtime bootstrap does not compile vLLM. Source build elapsed about 1484.674366 seconds.

Runtime-v3 contains the exact metadata dependency closure, the P31 native wheel, manifests, SHA256 checks, and compact evidence. `nvidia-cuda-nvdisasm 13.4.92` is retained because `nvidia-cutlass-dsl` requires `nvidia-cutlass-dsl-libs-base`, which requires nvdisasm `>=13.3,<14`. P31 recorded zero direct CUDA-13 ELF dependencies and zero CUDA-13 runtime contamination. This reachability does not establish a CUDA-13 runtime requirement.

Runtime-v3 archive SHA256: `0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73` (6,633,730,070 bytes; 215 members).

---

## Preserved historical v0.2 documentation

The section below is retained verbatim from the v0.2 repository state for historical users. Commands and defaults above describe the v0.3 candidate.

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
