from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

llm = LLM(
    model=MODEL,
    dtype="half",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.70,
    enforce_eager=True,
    disable_custom_all_reduce=True,
)

params = SamplingParams(temperature=0.0, max_tokens=64)
out = llm.generate(
    ["Explain tensor parallel inference in three short bullet points."],
    params,
)
print(out[0].outputs[0].text)
