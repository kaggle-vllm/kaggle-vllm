# OpenAI-compatible serving

Serving remains an upstream vLLM feature. `kaggle-vllm` only validates settings,
constructs an argument list, and starts the process without `shell=True`.

```python
from kaggle_vllm.server import ServerConfig, serve

raise SystemExit(serve(ServerConfig(
    model="/kaggle/input/qwen2.5-3b-t4x2-sharded",
    served_model_name="qwen2.5-3b-kaggle-t4x2",
    load_format="sharded_state",
    tensor_parallel_size=2,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.70,
    host="127.0.0.1",
    port=8001,
)))
```

The equivalent CLI is in the main README. Binding to `127.0.0.1` is the safe
default. Exposing a server beyond the notebook is an explicit user decision and
requires appropriate authentication/network controls outside this SDK.

The recorded Qwen server identified vLLM
`0.18.2.dev0+ga26e8dc7f.d20260822`, loaded the model with
`load_format=sharded_state` and TP=2, selected `TRITON_ATTN`, and returned:

- `GET /v1/models` — HTTP 200
- `POST /v1/chat/completions` — HTTP 200

These endpoint results establish functional compatibility in that session, not
capacity, latency, security, or production-readiness claims.
