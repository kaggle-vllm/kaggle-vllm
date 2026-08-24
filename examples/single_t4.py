"""Validated conservative single-T4 smoke configuration."""

from kaggle_vllm import KaggleLLM
from vllm import SamplingParams


llm = KaggleLLM(
    model="facebook/opt-125m",
    tensor_parallel_size=1,
    max_model_len=512,
    gpu_memory_utilization=0.50,
)
outputs = llm.generate(
    ["CUDA is a parallel computing platform developed by"],
    SamplingParams(temperature=0.0, max_tokens=24),
)
print(outputs[0].outputs[0].text)
