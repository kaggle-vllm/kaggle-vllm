"""Validated conservative dual-T4 tensor-parallel smoke configuration."""

from kaggle_vllm import KaggleLLM
from vllm import SamplingParams


llm = KaggleLLM(
    model="facebook/opt-125m",
    tensor_parallel_size=2,
    max_model_len=512,
    gpu_memory_utilization=0.40,
)
outputs = llm.generate(
    ["NCCL is used by distributed GPU applications to"],
    SamplingParams(temperature=0.0, max_tokens=32),
)
print(outputs[0].outputs[0].text)
