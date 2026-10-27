# Active compatibility target

| Dimension | Python 3.13 target |
|---|---|
| SDK version | 0.2.1 |
| Platform | Kaggle Linux x86_64 |
| Python ABI | CPython 3.13 / cp313 |
| Torch | observed 2.11.0+cu128 |
| CUDA | 12.8 family |
| GPUs | exactly two NVIDIA Tesla T4 / SM75 |
| TP | 1 or 2 |
| Upstream source identity | vLLM v0.18.1, `a26e8dc7ff2111a005144d775ecf9cebf56c45b2` |
| Native runtime | pending Kaggle build and validation |

The Python SDK import and CPU tests can pass without native vLLM. Native imports, NCCL, inference, Qwen loading, attention selection, and serving are unvalidated for cp313. [Historical cp312 acceptance](kaggle-v0.2.0-final-acceptance.md) does not prove cp313 compatibility.
