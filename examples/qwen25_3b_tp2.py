"""Qwen2.5-3B-Instruct FP16 TP=2 workflow validated on two Tesla T4s."""

from kaggle_vllm import KaggleLLM
from vllm import SamplingParams


llm = KaggleLLM(
    model="Qwen/Qwen2.5-3B-Instruct",
    tensor_parallel_size=2,
    max_model_len=2048,
    gpu_memory_utilization=0.70,
)
outputs = llm.generate(
    ["Explain tensor parallel model sharding."],
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(outputs[0].outputs[0].text)
