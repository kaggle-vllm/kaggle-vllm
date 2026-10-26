# Kaggle Python 3.13 dual-T4 target

The observed target audit reported Python 3.13.15, two Tesla T4 GPUs at SM75, usable PyTorch CUDA, PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0, Transformers 5.16.1, and protobuf 5.29.5. vLLM was absent. Simple FP32 and FP16 CUDA tests passed on both GPUs, but native vLLM imports, NCCL, TP=1/TP=2 inference, Qwen, and serving were not tested in that audit.

The active profile therefore remains `target_unvalidated`. See [compatibility](compatibility.md) and [native rebuild preparation](python313-native-runtime-rebuild.md).
