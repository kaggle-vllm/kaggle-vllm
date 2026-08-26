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
