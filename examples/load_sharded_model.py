"""Load the persistent TP-aware checkpoint with its validated TP topology."""

from vllm import SamplingParams

from kaggle_vllm import KaggleLLM

llm = KaggleLLM(
    model="/kaggle/input/qwen2.5-3b-t4x2-sharded",
    load_format="sharded_state",
    tensor_parallel_size=2,
    max_model_len=2048,
    gpu_memory_utilization=0.70,
)
outputs = llm.generate(
    ["What role does NCCL play in vLLM multi-GPU inference?"],
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(outputs[0].outputs[0].text)
