"""Run upstream vLLM's OpenAI-compatible server for the TP=2 checkpoint."""

from kaggle_vllm.server import ServerConfig, serve

config = ServerConfig(
    model="/kaggle/input/qwen2.5-3b-t4x2-sharded",
    served_model_name="qwen2.5-3b-kaggle-t4x2",
    load_format="sharded_state",
    tensor_parallel_size=2,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.70,
    host="127.0.0.1",
    port=8001,
)
raise SystemExit(serve(config))
