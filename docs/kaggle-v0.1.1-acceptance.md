# Kaggle dual-T4 v0.1.1 acceptance

`kaggle-vllm` 0.1.1 completed a fresh acceptance run on 2026-08-25 after its
PyPI publication and the Hugging Face repository renames. The source of truth
is the executed
[`kaggle_vllm_0_1_1_dual_t4_acceptance.ipynb`](../kaggle-notebooks/kaggle_vllm_0_1_1_dual_t4_acceptance.ipynb).
All substantive code cells executed, and the notebook contains no code-cell
error output.

This is configuration-specific functional evidence, not a production-readiness
or performance claim.

## Environment

| Component | Notebook observation |
|---|---|
| SDK | `kaggle-vllm==0.1.1`, installed from PyPI |
| Platform | Kaggle Linux, glibc 2.35 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA | 12.8 |
| CUDA toolkit | 12.8.93 |
| GPUs | 2 × NVIDIA Tesla T4 |
| Compute capability | 7.5 / SM75 on both GPUs |
| NCCL | 2.27.5 |

PyTorch remained at its existing Kaggle system path and version after SDK
installation and bootstrap. The lightweight SDK did not replace it.

## Native bootstrap identity

Strict profile validation reported `compatible: True` for
`kaggle-t4x2-cu128`. The actual bootstrap then completed successfully using:

| Field | Validated identity |
|---|---|
| Repository | `waqasm86/kaggle-vllm-binaries` |
| Immutable revision | `f6b4f10de54924ed6fe9e28cceab84eca7276ab6` |
| Wheel | `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl` |
| SHA256 | `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c` |

The activated package came from the notebook-owned staged runtime and reported
vLLM `0.18.2.dev0+ga26e8dc7f.d20260822`. Imports of `vllm`, `vllm._C`,
`vllm._moe_C`, and `vllm.cumem_allocator` all succeeded from that staged
location.

The acceptance run used `/kaggle/working/kaggle-vllm-e2e-011` for its staged
wheel, overlay, and manifest, plus the reusable
`/kaggle/working/kaggle-vllm-cache`. This clean-root pattern resolved an earlier
safe refusal caused by a non-empty default destination. The bootstrap refusal
is intentional and was not weakened.

## Tensor-parallel inference

The `facebook/opt-125m` smoke test initialized two workers with `world_size=2`,
NCCL, and TP ranks 0 and 1 on the two Tesla T4 GPUs. Real text generation
completed successfully with `tensor_parallel_size=2`.

Tesla T4 is SM75, so FlashAttention 2's compute-capability message is expected.
vLLM selected `TRITON_ATTN`. SymmMem communicator capability warnings were
also expected on SM75; ordinary NCCL TP=2 operation still completed.

## Persistent Qwen sharded state

The notebook freshly downloaded `waqasm86/kaggle-vllm-models`. The SDK shard
inspection reported:

| Field | Result |
|---|---|
| Valid | `True` |
| TP ranks | 2 |
| Rank-specific shards | 4 |
| Weight bytes | 6,172,262,512 (about 5.75 GiB) |

The copied original Hugging Face weight-index warning is expected: it names the
original Transformers shards, whereas the validated vLLM `sharded_state`
loader consumes the rank-specific files. This is not a standard Transformers
checkpoint.

The model resolved as `Qwen2ForCausalLM`, loaded with
`load_format="sharded_state"` and `tensor_parallel_size=2`, and completed real
text generation. The checkpoint remains topology-aware and is validated only
for TP=2; no arbitrary or uneven splitting claim is made.

## Observed resource footprint

The notebook's final GPU reading was 10,733 MiB used on each Tesla T4,
approximately 10.5 GiB per GPU, with `gpu_memory_utilization=0.70`. This setting
is a validated acceptance value, not a universal optimum.

The Kaggle UI observations supplied with the acceptance record were
approximately 2.2 GiB for the native bootstrap/runtime and approximately 8 GiB
of total working/output usage after the full Qwen flow. The exact Qwen weight
total above comes from the notebook. Cache state, package metadata, and
filesystem accounting can change overall storage, so these figures are
reference observations rather than guaranteed requirements.

No throughput, latency, or tokens-per-second benchmark is claimed by this run.
