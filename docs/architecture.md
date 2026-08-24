# Architecture

`kaggle-vllm` keeps policy and diagnostics separate from upstream execution.

```mermaid
flowchart LR
    U[User / CLI] --> R[Runtime diagnostics]
    U --> I[Explicit artifact staging]
    U --> K[KaggleLLM wrapper]
    K --> V[upstream vllm.LLM]
    V --> T[Single GPU or TP=2 execution]
    V --> S[persistent sharded_state]
    U --> O[ServerConfig]
    O --> C[vllm serve argument array]
```

The package does not import vLLM for diagnostics, install anything at import
time, implement kernels, replace schedulers, or reproduce vLLM inference.

## Modules

- `environment` collects a secret-free runtime fingerprint using lazy Torch
  discovery.
- `runtime` validates visible GPU count, Tesla T4 identity, SM75, and TP degree.
- `doctor` compares the runtime with the exact validated profile.
- `checksums` streams large-file SHA256 calculations.
- `installation` performs opt-in wheel/overlay staging while blocking Torch in
  overlay requirements.
- `llm` supplies conservative defaults and delegates to upstream `vllm.LLM`.
- `sharding` delegates saving and inspects rank/part topology without loading
  safetensor bodies.
- `server` builds and executes an argument array for upstream `vllm serve`.
- `cli` exposes the small operational surface.

The SDK is lightweight (`dependencies = []`). vLLM, Torch, CUDA, NCCL, and the
runtime overlay are environment responsibilities because blindly resolving
them would defeat the compatibility strategy.
