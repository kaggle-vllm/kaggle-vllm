# Four-L4 tensor parallelism

The L4 v0.3.0 profile requires exactly four visible NVIDIA L4 devices reporting compute capability 8.9 (SM89). `KaggleLLM` defaults to tensor parallel size 4. `kaggle-vllm verify-gpus --profile kaggle-l4x4-cu129` checks the exact topology. P31 NCCL world_size=4 passed, and Gemma 4 31B QAT loaded and generated meaningful chat output at TP=4.

Compatibility defaults are `enforce_eager=True` and `disable_custom_all_reduce=True`. They record a validated starting point, not a throughput optimum. Upstream vLLM still controls inference and model-specific attention backend selection. The validated Gemma backend was `TRITON_ATTN`.

---

## Preserved historical v0.2 documentation

The section below is retained verbatim from the v0.2 repository state for historical users. Commands and defaults above describe the v0.3 candidate.

# Multi-GPU, tensor parallelism and NCCL

Runtime tensor parallelism partitions upstream vLLM execution across a TP
group. `kaggle-vllm` validates the requested degree and passes it through; it
does not implement tensor partitioning.

```python
llm = KaggleLLM(model="facebook/opt-125m", tensor_parallel_size=2)
```

Historical evidence progressed from a raw two-process NCCL all-reduce, through
OPT-125M `world_size=2`, to Qwen2.5-3B TP=2 generation and a fresh-engine
reload of persistent sharded state. Both ranks returned the expected NCCL
result.

Persistent sharding is different: `save_sharded_state` writes topology-aware,
rank-specific files for later reload. A checkpoint saved at TP=2 should be
inspected and reloaded at TP=2. See [sharded state](sharded-state.md).

## SM75 behavior

Tesla T4 is compute capability 7.5 / SM75. Logs showed FlashAttention 2 was not
selected because it requires newer compute capability, then selected
`TRITON_ATTN`. SymmMem communicator optimizations were also unavailable.
Those optional optimization warnings did not prevent NCCL TP=2 inference:

```text
optional optimization unavailable != vLLM unsupported
```

## Performance boundary

TP=2 is not guaranteed to beat TP=1. Small-model results may be consistent with
collective/synchronization overhead outweighing partitioned compute, but
end-to-end throughput alone does not isolate that cause. Eager mode and
disabled custom all-reduce are compatibility-first historical defaults, not
universal performance recommendations. In the executed Milestone 1 OPT-125M
control, TP=2 output throughput was 26.73% lower in graph mode and 44.78% lower
in eager mode than TP=1. `nvidia-smi topo -m` reported `PHB` with no NVLink
token, while NVIDIA P2P read/write topology queries reported `OK`. These
observations do not isolate PCIe or NCCL as the sole cause. Use the controlled
[benchmark methodology and evidence](benchmarking.md).
