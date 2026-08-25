# kaggle-vllm v0.1.1

`kaggle-vllm` 0.1.1 is the lightweight SDK release freshly
acceptance-validated on Kaggle's dual NVIDIA Tesla T4 environment. It remains
an independent compatibility and packaging project around upstream vLLM, not
an official upstream vLLM release artifact or a production-readiness claim.

## Distribution

- [PyPI SDK](https://pypi.org/project/kaggle-vllm/)
- [GitHub source](https://github.com/kaggle-vllm/kaggle-vllm)
- [Native vLLM wheel and metadata](https://huggingface.co/waqasm86/kaggle-vllm-binaries)
- [Qwen TP=2 persistent sharded state](https://huggingface.co/waqasm86/kaggle-vllm-models)

The PyPI package is the small Python SDK. Native installation remains the
explicit, checksum-verified `kaggle-vllm bootstrap` operation; normal SDK
installation does not replace Kaggle's PyTorch stack.

## Fresh 2026-08-25 acceptance

The [executed acceptance notebook](https://github.com/kaggle-vllm/kaggle-vllm/blob/v0.1.1/kaggle-notebooks/kaggle_vllm_0_1_1_dual_t4_acceptance.ipynb)
and [evidence report](https://github.com/kaggle-vllm/kaggle-vllm/blob/v0.1.1/docs/kaggle-v0.1.1-acceptance.md)
record:

- Python 3.12.13, PyTorch 2.10.0+cu128, CUDA toolkit 12.8.93
- 2 × Tesla T4 at SM75 with NCCL 2.27.5
- strict profile compatibility and successful native bootstrap
- staged `vllm`, `vllm._C`, `vllm._moe_C`, and allocator imports
- real `facebook/opt-125m` TP=2 generation
- valid inspection of the canonical Qwen repository: two ranks, four shards
- Qwen `sharded_state` TP=2 load and successful generation
- expected SM75 fallback to `TRITON_ATTN`; SymmMem warnings did not prevent
  ordinary NCCL TP=2 operation

## Immutable native runtime

- Repository: `waqasm86/kaggle-vllm-binaries`
- Revision: `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`
- Wheel: `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`
- SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

No native wheel was rebuilt, no Qwen checkpoint was regenerated, and no large
runtime or model artifact is duplicated in this GitHub release.
