# Research-tooling feasibility

## GuideLLM

The supplied checkout is commit
`fc2dbe9edd4f7f1a4e9ccd752f6f43591adbcb73`, described as
`v0.7.3-47-gfc2dbe9e`. Source inspection found OpenAI-compatible endpoint load,
synthetic token workloads, concurrent request streams, and JSON/CSV outputs.
Integration is feasible as an optional M4 cross-check. Its Torch, Transformers,
NumPy and related dependency set is too large for the lightweight SDK; run it
as isolated research tooling. The prepared notebook creates a separate client
virtual environment and starts the canonical vLLM server from its unchanged
runtime. The exact source-backed CLI uses `openai_http`, `/v1/completions`,
`synthetic_text`, a pinned Hugging Face tokenizer revision, concurrent streams,
static seeds, max-request/error constraints, and JSON/CSV outputs. Preserve
both tools' metric definitions before comparing TTFT, TPOT, ITL, latency, or
throughput. Current status: `PREPARED_FOR_KAGGLE`, not measured.

## NVIDIA nsight-python

The supplied checkout is commit
`b061ae742867a7ef551aba2508bd5c39d043eea6`. Its inspected requirements target `ncu-report>=2026.2.0`, Nsight
Compute with CUDA 13.3 or newer, and CUDA Python bindings at least 12.9.6 (with
13.0/13.1 exclusions). This host reports `ncu` 2025.1.1 and the canonical
Kaggle environment is CUDA toolkit 12.8. Installing it there could perturb the
validated runtime, so it is deliberately excluded. Use CUDA events, PyTorch
profiler, NVTX, NCCL logging and nvidia-smi/NVML in canonical runs. Any later
Nsight run belongs in an isolated auxiliary environment and cannot replace the
canonical measurements.

## Microsoft Vidur / M5

A complete workspace path/name search found no supplied Vidur source tree or
archive. Because the task requires inspection of the exact supplied source,
no network substitute was fetched and no APIs, T4 network profile, compute
profile, scheduler mapping or simulated result was invented. Current status:
`SIMULATOR_COMPATIBILITY_LIMITATION`. Once the intended source is supplied,
inspect its model configuration, compute/network profilers, scheduler and CPU
overhead assumptions before deciding whether `t4_pair_phb` is defensible.
Calibration observations must remain separate from independent validation.
GuideLLM may satisfy the independent-validation gate if repeated real results
pass audit; that does not turn the absent simulator into a successful result.
