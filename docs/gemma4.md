# Gemma 4 31B QAT example

Only `gemma-4-31b-it-qat-w4a16-ct` on Kaggle four L4 GPUs at TP=4 has the stated semantic qualification. The model remains a Kaggle Input; it is not packaged. `discover_gemma4_qat()` chooses the highest numbered version directory with `config.json` and rejects ambiguous or missing inputs.

```python
from kaggle_vllm import KaggleLLM, KaggleSamplingParams, activate_runtime, discover_gemma4_qat

activate_runtime()
llm = KaggleLLM(model=discover_gemma4_qat(), profile="kaggle-l4x4-cu129", dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.86)
outputs = llm.chat(
    [{"role": "user", "content": "Explain tensor parallelism in one sentence."}],
    sampling_params=KaggleSamplingParams(temperature=0.0, max_tokens=64),
)
```

`KaggleLLM.chat()` renders a chat template with `AutoProcessor` and falls back to `AutoTokenizer`. The wrapper forwards upstream options and exceptions. TP=4 is the hardware default for this profile. `dtype="bfloat16"` is a validated Gemma model choice. `enforce_eager=True` and `disable_custom_all_reduce=True` are compatibility defaults and can be overridden. Upstream chooses the attention backend; the qualified Gemma run used `TRITON_ATTN`, while the separate built-in FA2 extension imported successfully.
