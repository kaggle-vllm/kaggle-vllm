# Frequently asked questions

## Why not `pip install vllm`?

Normal dependency resolution can replace Kaggle's coordinated Torch/CUDA
stack. This project stages one verified native wheel and a reviewed overlay
without dependencies, then activates them explicitly.

## Is this a vLLM fork?

No. It is compatibility, delivery and wrapper code around upstream vLLM.

## Does the SDK support Python 3.10–3.13?

The lightweight SDK is CPU-tested across those versions. The current native
wheel is cp312 and works only with the Python 3.12 native profile.

## Is CUDA “13.0” the T4 compute capability?

No. `nvidia-smi` reported the driver's maximum CUDA compatibility as 13.0.
Tesla T4 compute capability is 7.5 / SM75; the built toolkit and Torch ABI are
CUDA 12.8.

## Is TP=2 faster?

Not proven. For small models, NCCL communication can outweigh compute savings.
Run and review the pending benchmark matrix.

## Can Transformers load the Qwen artifact?

Not as a normal checkpoint. It is upstream vLLM persistent `sharded_state`
with rank-specific TP=2 files and a stale copied Transformers index.

## Is the Qwen artifact Apache-2.0 or commercially unrestricted?

No. Only the SDK is Apache-2.0. The Qwen weights remain subject to the included
Qwen Research License and require a separate commercial license.

## Does a FlashAttention warning mean vLLM failed on T4?

No. FlashAttention 2 was not selected on SM75; `TRITON_ATTN` and NCCL TP=2
completed successfully in the recorded configuration.
