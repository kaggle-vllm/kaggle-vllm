# Compatibility matrix

| Dimension | Validated | Outside evidence / limitation |
|---|---|---|
| Host | Kaggle Notebook, glibc 2.35 | Other hosts not validated |
| Lightweight SDK | Python 3.10+; locally tested on 3.11 | Does not itself provide CUDA/vLLM |
| Native profile | CPython 3.12.13, wheel ABI cp312 | Bootstrap rejects cp310/cp311/cp313 |
| PyTorch | 2.10.0+cu128 | Other Torch ABIs not validated |
| CUDA | toolkit 12.8.93 | Other toolkit builds not validated |
| GPU | 1 or 2 Tesla T4 for tested paths | Other GPUs not validated by this evidence |
| Compute | SM75 | FA2 unavailable; TRITON_ATTN selected |
| NCCL | 2.27.5 | Version changes not validated |
| Precision | FP16 | BF16 is not a T4 default; quantization not tested |
| Runtime TP | TP=1 and TP=2 | TP>2 and multi-node not tested |
| Qwen model | Qwen2.5-3B-Instruct, TP=2 | Other model/topology pairs not validated |
| Persistence | vLLM `sharded_state`, TP=2 | Standard Transformers loading not supported |
| Serving | Local OpenAI models/chat endpoints | Load/performance/security not benchmarked |
| Delivery | HF commit pinned + SHA256 verified | Mutable `main` is not the bootstrap source |

Compatibility here means "functionally validated on the documented Kaggle
dual-T4 environment," not a general support certification.
