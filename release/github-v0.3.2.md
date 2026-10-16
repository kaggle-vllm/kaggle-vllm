# kaggle-vllm v0.3.2 — release draft

This pure Python SDK release candidate prepares a versioned public runtime for exactly four NVIDIA L4 GPUs (SM89), CPython 3.12, CUDA 12.9 user-space libraries, Torch 2.13.0+cu129 and upstream vLLM 0.30.0. Tensor parallel size is 4. The vLLM native wheel remains the exact prior 293,855,397-byte artifact with SHA256 `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`.

The historical 0.2.0 T4x2/SM75/cu128/TP2 release remains available. Its native wheel and pinned Hugging Face commit are unchanged.

## Installation after public artifact completion

```bash
python -m pip install "kaggle-vllm==0.3.2"
kaggle-vllm network-preflight
kaggle-vllm bootstrap --profile kaggle-l4x4-cu129 --strict --json
eval "$(kaggle-vllm env --profile kaggle-l4x4-cu129)"
kaggle-vllm doctor --profile kaggle-l4x4-cu129 --strict --json
```

The large dependency wheelhouse is distributed through `waqasm86/kaggle-vllm-binaries` on Hugging Face. The immutable revision will be inserted only after all 168 locked dependency wheels, native wheel and metadata are publicly present and verified. The SDK refuses mutable revisions or incomplete artifact sets.

## Status and limitations

- Local Python 3.11 CPU suite: 425 passed, 1 skipped. The pure Python wheel and sdist built and passed Twine checks.
- Four L4 GPU acceptance for the exact 0.3.2 SDK: **NOT RUN**. Native import, TP4, Gemma 4 semantic generation and cleanup require the user's manual Kaggle notebook.
- P31 historical Gemma 4 generation used `TRITON_ATTN`. A compiled FlashAttention 2 extension import does not prove Gemma 4 FA2 execution or optimized throughput.
- Kaggle Internet must be enabled for PyPI and Hugging Face downloads. DNS errors are network failures; the SDK does not bypass Kaggle restrictions.
- Third-party binary redistribution notices and rights are under review. The public wheelhouse must not be uploaded until cleared.

Attach the reviewed SDK wheel, source distribution and `release/v032-SHA256SUMS.txt` to the GitHub release. Keep large native/runtime dependencies on Hugging Face. Include `LICENSE`, staged notices, build provenance and this release note. Do not publish this draft or claim production readiness before the remaining gates pass.
