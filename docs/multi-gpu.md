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
measured optima. Use the controlled [benchmark methodology](benchmarking.md).
