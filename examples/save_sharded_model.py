"""Create a persistent vLLM-native TP=2 sharded_state checkpoint."""

from pathlib import Path

from huggingface_hub import snapshot_download

from kaggle_vllm import KaggleLLM

model_path = Path(snapshot_download("Qwen/Qwen2.5-3B-Instruct"))
llm = KaggleLLM(
    model=model_path,
    tensor_parallel_size=2,
    max_model_len=2048,
    gpu_memory_utilization=0.70,
)
inspection = llm.save_sharded_model(
    "/kaggle/working/qwen2.5-3b-t4x2-sharded",
    max_size=2 * 1024**3,
    metadata_source=model_path,
)
print(inspection.to_dict())
