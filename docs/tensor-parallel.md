# Tensor parallel inference

Tensor parallelism partitions execution across a TP group at runtime. It does
not create a reusable on-disk split by itself.

```python
from kaggle_vllm import KaggleLLM

llm = KaggleLLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    tensor_parallel_size=2,
    max_model_len=2048,
    gpu_memory_utilization=0.70,
)
```

The wrapper refuses TP=2 when PyTorch sees only one device. It passes the TP
degree to upstream vLLM and does not implement tensor partitioning.

## Evidence progression

1. A raw two-process NCCL all-reduce returned `3.0` at both ranks.
2. `facebook/opt-125m` initialized with world size two, TP ranks 0 and 1, then
   completed generation.
3. Qwen/Qwen2.5-3B-Instruct initialized in FP16 at TP=2 and generated output.
4. A second engine loaded its persistent TP=2 `sharded_state` and generated.
5. The upstream OpenAI server served that sharded model at TP=2.

The settings `dtype="float16"`, `enforce_eager=True`, and
`disable_custom_all_reduce=True` reduce variables for this tested T4 topology.
They are conservative validation defaults, not universal performance advice.
No benchmark established that they are optimal.
