# Persistent TP-aware sharded state

The experiment called vLLM's native engine-core `save_sharded_state` at TP=2
with a 2 GiB part limit. Each TP rank wrote two safetensors, producing four
rank-specific files and 6,172,262,512 weight bytes in total.

```python
inspection = llm.save_sharded_model(
    "/kaggle/working/qwen2.5-3b-t4x2-sharded",
    max_size=2 * 1024**3,
)
```

For a local HF snapshot, the wrapper also copies non-weight model/tokenizer
metadata. It refuses to overwrite a non-empty destination.

Inspect without loading tensor bodies:

```python
from kaggle_vllm import inspect_sharded_model

result = inspect_sharded_model("/kaggle/input/qwen2.5-3b-t4x2-sharded")
assert result.rank_count == 2
```

Reload:

```python
from kaggle_vllm import KaggleLLM

llm = KaggleLLM(
    model="/kaggle/input/qwen2.5-3b-t4x2-sharded",
    load_format="sharded_state",
    tensor_parallel_size=2,
    max_model_len=2048,
    gpu_memory_utilization=0.70,
)
```

The extracted, directly usable TP=2 repository is published at
[waqasm86/vllm-kaggle-models](https://huggingface.co/waqasm86/vllm-kaggle-models).
Pass that repository ID directly as `model` with `load_format="sharded_state"`
and `tensor_parallel_size=2`. The repository is not a newly trained or
fine-tuned model and does not contain new learned weights.

## Index-file caveat

The archive preserved the original HF `model.safetensors.index.json`. Its 434
weight mappings name `model-00001-of-00002.safetensors` and
`model-00002-of-00002.safetensors`, which are intentionally absent. The
validated vLLM `sharded_state` loader uses `model-rank-{rank}-part-{part}` files
and loaded successfully. `inspect_sharded_model` warns about this distinction.

This layout should normally be loaded with the TP topology for which it was
generated. It is not a standard Transformers checkpoint, arbitrary tensor
split, uneven GPU memory scheme, or topology portability guarantee.
