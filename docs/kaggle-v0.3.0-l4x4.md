# v0.3.0 L4×4 local release candidate

This profile is `kaggle-l4x4-cu129`: Kaggle Linux x86_64, CPython 3.12/cp312, exactly four NVIDIA L4 GPUs at compute capability 8.9 (SM89), TP=4, upstream vLLM 0.30.0, PyTorch 2.13.0+cu129, CUDA runtime 12.9, Triton 3.7.1, TorchCodec 0.16.0+cu129, FlashInfer 0.6.18.post1, and NCCL 2.29.7. Strict validation rejects T4, any non-L4 name, any GPU count other than four, wrong compute capability or ABI, and a different native-wheel hash.

The authoritative native wheel is `vllm-0.30.0-cp312-cp312-linux_x86_64.whl`, 293855397 bytes, SHA256 `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`. The older P29 wheel hash `abc3771627f2576579928e55ef970a3282cb72a3d6336ea4f07173c8bb687ec0` is not accepted. The SDK contains metadata, not the 294 MB wheel.

P31/P33/P35/P40 establish native import, four-rank NCCL, and Gemma 4 31B QAT TP=4 semantic inference. The native wheel contains an importable built-in FA2 extension. Gemma selected `TRITON_ATTN` because the tested architecture had heterogeneous attention head dimensions. The SDK does not force an attention backend or claim every model uses FA2.

The accepted SDK candidate wheel (SHA256 `7da66878a85c3b65db18d794d282519d853893f54d7314db5ae551ace377416b`) passed all 19 mandatory cleanup-r1 Kaggle gates. This feature branch has source changes, so its rebuilt SDK wheel requires a separate fresh Kaggle run. See the [acceptance audit](release/v0.3.0-acceptance-audit.md).

The native build used `CMAKE_CUDA_ARCHITECTURES=89`, `TORCH_CUDA_ARCH_LIST=8.9`, and the output of `nproc --all` for build concurrency. One session returned 48; this is not a universal CPU requirement. Normal SDK bootstrap does no compilation.

## Review CLI

`kaggle-vllm profile` lists the two packaged profiles. Use `kaggle-vllm plan --profile kaggle-l4x4-cu129 --runtime-source /path/to/offline/inputs --native-wheel /path/to/exact/native.whl` to preview the selected runtime, and `kaggle-vllm verify-hashes --native-wheel /path/to/exact/native.whl --runtime-source /path/to/offline/inputs` for a full checksum pass. The latter reads the 168 pinned runtime wheels and may take several minutes. `kaggle-vllm discover-model --root /kaggle/input/models` lists local model directories with `config.json`. `kaggle-vllm evidence-export --profile kaggle-l4x4-cu129 --output /kaggle/working/doctor.json --strict` writes a doctor report once to a new file. The production acceptance runner creates the full 19-gate evidence archive.

For Python inference, pass `profile="kaggle-l4x4-cu129"` and `tensor_parallel_size=4` explicitly. Local calls without a profile retain the v0.2.0 T4 wrapper defaults, including TP=1. Explicit T4 profile calls default to TP=2. Select a profile before loading upstream vLLM; an active L4 overlay cannot be switched to T4 in the same Python process.
