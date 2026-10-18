# kaggle-vllm: clean CPython 3.12 development rebuild

**Status: development scaffold, not GPU-certified.**

This is a fresh SDK source tree for:

- 4× NVIDIA L4 (SM89)
- CPython 3.12 at `/usr/bin/python3.12`
- Native `vllm==0.30.0`
- `torch==2.13.0+cu129`
- `triton==3.7.1`
- Offline Kaggle Inputs

The native vLLM wheel and its dependencies must be
installed independently and verified before importing vLLM.

The SDK does not download, compile or redistribute CUDA code.

## Important Python kernel limitation

An SDK cannot apply `%%script /usr/bin/python3.12`
automatically across notebook cells.

Run your workload through an explicit Python 3.12
subprocess, or select a genuine Python 3.12 Jupyter kernel.

## Publication

This is `0.3.0.post1.dev0` while under development.
It does not replace the previously published 0.3.0 wheel.
