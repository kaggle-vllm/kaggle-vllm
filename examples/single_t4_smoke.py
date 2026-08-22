from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

llm = LLM(
    model=MODEL,
    dtype="half",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.70,
    enforce_eager=True,
)

params = SamplingParams(temperature=0.0, max_tokens=32)
out = llm.generate(["What is CUDA? Answer in one sentence."], params)
print(out[0].outputs[0].text)
