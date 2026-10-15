# Troubleshooting

- **Python ABI mismatch:** The SDK can be developed under Python 3.11, but its native vLLM wheel is cp312. Use a Kaggle CPython 3.12 session for bootstrap and inference.
- **Wrong GPU topology:** Strict mode requires exactly four NVIDIA L4/SM89 devices. A T4, A100, P100, or smaller L4 allocation is outside v0.3.0.
- **Runtime input not found:** Attach corrected split-v2 Dataset Inputs and pass `--runtime-source /kaggle/input/datasets/waqasm86`, or use deterministic discovery with `--offline`. `--runtime-root` chooses the expanded scratch destination.
- **FlashInfer JIT `-lcuda` failure:** Run bootstrap and inspect `kaggle-vllm doctor --json`; it creates an SDK-owned linker alias to the live driver and records an optional g++ probe.
- **TorchCodec requests `libnvrtc.so.13`:** That symptom belonged to the old TorchCodec 0.17.0 attempt. Use corrected runtime-v3 with `0.16.0+cu129`.
- **Gemma emits newline-like output:** Raw prompts did that in P31 Step 17. Use `llm.chat()` or render the chat template with `AutoProcessor`. P31 Step 20 semantic chat passed.
- **FA2 extension imports but Gemma uses Triton:** Expected. Extension availability and model backend selection are separate checks.

---

## Preserved historical v0.2 documentation

The section below is retained verbatim from the v0.2 repository state for historical users. Commands and defaults above describe the v0.3 candidate.

# Troubleshooting

## CMake cannot create `CUDA::cuda_driver`

If the CUDA toolkit is found but the imported driver target is absent on
Kaggle, expose the managed driver mount:

```bash
export CMAKE_LIBRARY_PATH=/usr/local/nvidia/lib64
```

The recorded live driver was present and functional. This is a search-path
layout issue, not “Kaggle has no NVIDIA driver.”

## `xgrammar` fails because `tvm_ffi` is missing

The successful overlay paired `xgrammar==0.2.3` with distribution
`apache-tvm-ffi==0.1.13.post3` (import `tvm_ffi`). Activate the completed
overlay, run `kaggle-vllm doctor --strict`, and confirm both are `PASS`. Do not
solve this by letting pip replace the entire Torch/vLLM stack.

## FlashAttention 2 or SymmMem warning on T4

SM75 did not select FlashAttention 2 and did not support the observed SymmMem
optimization. `TRITON_ATTN` was selected and real NCCL TP=2 inference still
passed. Treat the exact known warning as non-fatal unless followed by a real
initialization/generation failure.

## Bootstrap refuses a non-empty destination

This is a safety property. Inspect ownership and the exact reset plan:

```bash
kaggle-vllm bootstrap --reset-runtime --dry-run --strict
```

If the plan proves manifest ownership, explicitly confirm:

```bash
kaggle-vllm bootstrap --reset-runtime --yes --strict
```

The checksum-verified cache remains preserved. Never bypass the refusal with a
broad recursive delete.

## Native extension cannot find Torch/CUDA libraries

Use `kaggle-vllm env` or `activate_runtime()` so the staged wheel, overlay,
Torch `lib`, live driver and CUDA runtime paths are visible to the current
process. Confirm Torch still resolves to Kaggle's original system path. Do not
install another Torch as a shortcut.

## Qwen checkpoint does not load with Transformers

The artifact is vLLM native `sharded_state`, not a standard Transformers
checkpoint. Use `load_format="sharded_state"`, TP=2, and inspect with:

```bash
kaggle-vllm inspect-shards MODEL_DIR --tensor-parallel-size 2 --json
```

The retained upstream `model.safetensors.index.json` references original HF
shards that are absent. Rank-specific files are authoritative for this loader.
