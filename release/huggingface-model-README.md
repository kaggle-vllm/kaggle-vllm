---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: vllm
tags:
- qwen2.5
- vllm
- sharded-state
- tensor-parallel
- kaggle
- tesla-t4
---

# Qwen2.5-3B-Instruct — vLLM TP=2 persistent sharded state

This is **not a newly trained or fine-tuned model**, and it does not introduce
new learned weights. It is a mechanically transformed, vLLM-native persistent
`sharded_state` representation of
[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct),
created and validated on a Kaggle Notebook with two Tesla T4 GPUs.

The representation was generated with vLLM tensor parallel size 2. It contains
rank-specific checkpoint files for TP ranks 0 and 1. It is not arbitrary tensor
splitting and should normally be loaded using the TP topology for which it was
generated.

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c)
> Alibaba Cloud. All Rights Reserved. The included license permits
> non-commercial research/evaluation use and redistribution subject to its
> terms. Commercial use requires a separate license from Alibaba Cloud.

## Load with kaggle-vllm

```python
from kaggle_vllm import KaggleLLM

llm = KaggleLLM(
    model="waqasm86/vllm-kaggle-models",
    load_format="sharded_state",
    tensor_parallel_size=2,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.70,
    enforce_eager=True,
    disable_custom_all_reduce=True,
)
```

## Load directly with upstream vLLM

```python
from vllm import LLM

llm = LLM(
    model="waqasm86/vllm-kaggle-models",
    load_format="sharded_state",
    tensor_parallel_size=2,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.70,
    enforce_eager=True,
    disable_custom_all_reduce=True,
)
```

## OpenAI-compatible server

```bash
vllm serve waqasm86/vllm-kaggle-models \
  --served-model-name qwen2.5-3b-kaggle-t4x2 \
  --load-format sharded_state \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.70 \
  --enforce-eager \
  --disable-custom-all-reduce
```

## Validation matrix

| Component | Validated value |
|---|---|
| Origin | Qwen/Qwen2.5-3B-Instruct |
| Representation | vLLM `sharded_state`, two ranks × two parts |
| Precision | FP16 runtime validation |
| Hardware | 2 × Tesla T4, SM75 |
| Python / Torch / CUDA | 3.12.13 / 2.10.0+cu128 / toolkit 12.8.93 |
| NCCL | 2.27.5 |
| vLLM source | v0.18.1, `a26e8dc7ff2111a005144d775ecf9cebf56c45b2` |
| vLLM wheel | `0.18.2.dev0+ga26e8dc7f.d20260822.cu128` |
| Reload | PASS at `load_format=sharded_state`, TP=2 |
| OpenAI API | models and chat completions returned HTTP 200 |

## Packaging notes and limitations

- Files were mechanically transformed by vLLM's native TP=2 checkpoint saver;
  this README and NOTICE were added for packaging/attribution.
- This layout is not a standard Transformers checkpoint. The retained upstream
  safetensors index references the original HF shard names; vLLM's validated
  `sharded_state` loader uses the rank-specific files. Do not expect
  `AutoModelForCausalLM.from_pretrained()` to load this repository as a normal
  Transformers checkpoint.
- TP=1, TP>2, uneven GPU splits, other topologies, other accelerators, and
  training/fine-tuning were not validated.
- FlashAttention 2 is unavailable on SM75; vLLM selected `TRITON_ATTN`.
- The conservative settings shown above are not claimed to be optimal.
- Use is limited by the included Qwen Research License. Retain `LICENSE` and
  `NOTICE` when redistributing.

Upstream attribution: Qwen Team / Alibaba Cloud for the model and license;
vLLM contributors for the Apache-2.0 inference and sharded-state machinery.
