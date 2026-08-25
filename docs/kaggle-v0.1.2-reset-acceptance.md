# Kaggle v0.1.2 focused runtime-reset acceptance

Validation date: 2026-08-25

Source evidence:
[`kaggle_vllm_0_1_2_reset_acceptance.ipynb`](../kaggle-notebooks/kaggle_vllm_0_1_2_reset_acceptance.ipynb)
(SHA256 `ba496eb0a1acf226741f32780b8156a57d7d2d855f21083e8a103000ea5a70b5`).
The executed notebook contains no Jupyter `error` outputs and ends with
`FINAL ACCEPTANCE: PASS`.

This was a focused acceptance of the filesystem-reset behavior added in
`kaggle-vllm==0.1.2`. It did not rebuild upstream vLLM, change the native
wheel, regenerate Qwen, or repeat the multi-GB Qwen download/reload.

## Observed environment

| Component | Saved notebook output |
|---|---|
| Platform | Kaggle Linux 6.12.90+, glibc 2.35 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA | 12.8 |
| CUDA toolkit | 12.8.93 |
| GPUs | 2 × Tesla T4 |
| Compute capability | 7.5 / SM75 on both GPUs |
| NCCL | 2.27.5 |
| SDK | PyPI `kaggle-vllm==0.1.2` |

The package installed from public PyPI and reported
`kaggle_vllm.__version__ == "0.1.2"`.

## Immutable native runtime

- Hugging Face repository: `waqasm86/kaggle-vllm-binaries`
- Immutable revision: `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`
- Wheel: `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`
- SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

The initial strict dry-run returned `compatible: true` for CPython 3.12,
Linux x86_64, the Kaggle runtime, Torch/CUDA, two T4/SM75 devices, and NCCL.
The first strict bootstrap then downloaded or reused the exact wheel, staged
the native runtime and locked overlay, and wrote its runtime manifest.

## Reset safety evidence

The completed runtime used notebook-owned paths under
`/kaggle/working/kaggle-vllm-e2e-012`, with the reusable cache at
`/kaggle/working/kaggle-vllm-cache`.

The notebook then exercised each safety state:

1. A normal bootstrap probe against the populated staged/overlay paths, but
   without their matching ownership manifest, returned code 2 with
   `refusing to overwrite non-empty staged wheel destination`. The notebook
   explicitly recorded this as the expected safety outcome.
2. `bootstrap --reset-runtime --dry-run` classified staged, overlay, and
   manifest as `remove` because they matched the runtime manifest, and the
   cache as `preserve`. Before/after filesystem snapshots were exactly equal.
3. `bootstrap --reset-runtime --yes --strict` validated the same ownership,
   removed only staged/overlay/manifest state, preserved the cache, and
   immediately completed a strict bootstrap again.
4. The staged directory, overlay, and manifest were recreated; the cache
   snapshot remained unchanged.

The final acceptance dictionary recorded all twelve required checks as true:

- `cache_preserved`
- `environment_dual_t4`
- `expected_non_empty_refusal`
- `explicit_reset`
- `initial_strict_bootstrap`
- `native_imports`
- `opt_tp2_generation`
- `pypi_0_1_2`
- `reset_dry_run_non_mutating`
- `strict_dry_run`
- `strict_rebootstrap`
- `torch_preserved`

## Native imports and Torch preservation

The recreated runtime reported vLLM
`0.18.2.dev0+ga26e8dc7f.d20260822`. Imports resolved from the recreated staged
runtime:

| Import | Saved path |
|---|---|
| `vllm` | `/kaggle/working/kaggle-vllm-e2e-012/vllm-staged/vllm/__init__.py` |
| `vllm._C` | `/kaggle/working/kaggle-vllm-e2e-012/vllm-staged/vllm/_C.abi3.so` |
| `vllm._moe_C` | `/kaggle/working/kaggle-vllm-e2e-012/vllm-staged/vllm/_moe_C.abi3.so` |
| `vllm.cumem_allocator` | `/kaggle/working/kaggle-vllm-e2e-012/vllm-staged/vllm/cumem_allocator.abi3.so` |

The notebook's `Torch before` and `Torch after` dictionaries were identical:
version `2.10.0+cu128`, CUDA `12.8`, and path
`/usr/local/lib/python3.12/dist-packages/torch/__init__.py`. Bootstrap did not
replace Kaggle's Torch installation.

## Real TP=2 inference

The `facebook/opt-125m` engine logs recorded `world_size=2`, rank 0 and rank 1
using `backend=nccl`, and assignments to TP rank 0 and TP rank 1. Generation
completed successfully for the prompt
`Kaggle dual NVIDIA T4 runtime reset acceptance is`, producing non-empty text
(` enabled.` in this run).

FlashAttention 2 unavailability on compute capability 7.5 was expected. vLLM
selected `TRITON_ATTN`. SymmMem also reported that device capability 7.5 is not
supported; this warning did not prevent ordinary NCCL TP=2 initialization or
generation.

## Resource observations

The notebook does not embed machine-readable Kaggle UI disk/RAM/VRAM
telemetry. The operator reported screenshot-time reference values of about
2.2 GiB output usage, 7.3 GiB RAM, and 9 GiB GPU memory per T4. These are
explicitly not notebook assertions, release gates, performance benchmarks, or
guaranteed package requirements.

## Scope

This evidence closes the focused v0.1.2 release gate: safe reset planning,
explicit reset, cache preservation, strict re-bootstrap, native imports, Torch
preservation, and short real TP=2 inference. Qwen was not freshly tested under
v0.1.2; its unchanged TP=2 `sharded_state` artifact remains covered by the
[v0.1.1 full acceptance](kaggle-v0.1.1-acceptance.md).
