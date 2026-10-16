# Native redistribution and provenance
`release/v032-redistribution-clearance.json` remains `BLOCKED`. The exact L4 wheel is already public at Hugging Face `v0.3.0/l4x4/native/vllm-0.30.0-cp312-cp312-linux_x86_64.whl`, current Hub commit `8d7dd28f7c76c3b68e7326df81758411b7dc2680`, with the expected LFS SHA256 and size. The SDK profile and revised upload plan reuse that historical object without copying it. This removes a proposed second redistribution, but existing publication alone does not settle original redistribution rights or notices.
The seven native `.so` objects, their dynamic libraries and three embedded license files are inventoried in `native-compiled-object-inventory.json`. The source commit, exact build inputs, bundled third-party code and notices have not been reconciled sufficiently to mark redistribution PASS. NVIDIA/CUDA terms require component-specific review. No Hub copy or new binary distribution was performed.

## Seven compiled objects: disposition for a new public copy

| Object | Candidate source and observed evidence | Missing evidence | Decision |
| --- | --- | --- | --- |
| `vllm/_C_stable_libtorch.abi3.so` | vLLM extension; wheel embeds vLLM Apache license; links Torch and CUDA runtime | Exact source/build mapping, third-party code and NVIDIA build terms | BLOCKED |
| `vllm/_moe_C_stable_libtorch.abi3.so` | vLLM MoE extension; links Torch and CUDA runtime | Exact kernel sources, included third-party code and notices | BLOCKED |
| `vllm/cumem_allocator.abi3.so` | vLLM allocator extension; links CUDA driver and Torch | Exact source/build mapping and NVIDIA terms | BLOCKED |
| `vllm/fs_io_C.abi3.so` | vLLM filesystem I/O extension; links Torch | Exact source/build mapping and included notices | BLOCKED |
| `vllm/spinloop.abi3.so` | vLLM spinloop extension; links Torch | Exact source/build mapping and included notices | BLOCKED |
| `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so` | FlashAttention extension; wheel embeds `vllm/vllm_flash_attn/cute/LICENSE`; links Torch/CUDA | Exact FlashAttention/CUTE/CUTLASS source revisions, license scope and notice fulfillment | BLOCKED |
| `vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so` | FlashAttention extension; same embedded CUTE license and Torch/CUDA linkage | Exact FlashAttention/CUTE/CUTLASS source revisions, license scope and notice fulfillment | BLOCKED |

These are cautious candidate identifications from names and ELF linkage, not a build-source attestation. The existing [GitHub v0.3.1 release](https://github.com/kaggle-vllm/kaggle-vllm/releases/tag/v0.3.1) exposes the exact 293,855,397-byte wheel with GitHub digest `sha256:9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`. The SDK and metadata now reference the exact historical Hub object to avoid a second copy. Existing distribution rights still require review.

The local build evidence at `../kaggle-vllm-0300/evidence/source-identity.json` records the Kaggle vLLM source delivery, release markers and hashes of six source/build files. `../kaggle-native-v2-upload/NATIVE_SOURCE_PROVENANCE.json` records commits for CUTLASS, FlashAttention and eight other auxiliary repositories; its `SHA256SUMS.txt` hashes their source archives. The 398,113,679-byte Kaggle-safe source ZIP contains 49,354 members and license files for CUTLASS, FlashAttention and auxiliary repositories. These records narrow the candidate build inputs, but do not prove which archive revisions or source objects produced each of the seven exact `.so` bytes. The wheel includes only three embedded license files. Required attribution and any NVIDIA component terms remain unresolved, so clearance stays BLOCKED.
