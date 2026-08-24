# kaggle-vllm v0.1.0-rc1

Research/compatibility prerelease. Functionally validated on the documented
Kaggle dual-T4 environment; not a production-readiness claim.

## Validated binary

- Upstream source tag: vLLM v0.18.1
- Upstream commit: `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`
- Generated distribution version:
  `0.18.2.dev0+ga26e8dc7f.d20260822.cu128`
- Wheel ABI: CPython 3.12 / Linux x86_64
- SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

The generated version is `setuptools_scm` distribution metadata from the
v0.18.1 source checkout; it is not a claim that the source was upstream v0.18.2.

## Hugging Face distribution

- [Validated vLLM wheel and compatibility metadata](https://huggingface.co/waqasm86/vllm-kaggle-binaries)
- [Qwen2.5-3B-Instruct TP=2 persistent sharded state](https://huggingface.co/waqasm86/vllm-kaggle-models)

The Qwen repository is a vLLM-native topology-aware representation, not a new
trained/fine-tuned model or a standard Transformers checkpoint.

## Environment and functional validation

- Python 3.12.13, PyTorch 2.10.0+cu128, CUDA toolkit 12.8.93
- NVIDIA driver 580.159.04, NCCL 2.27.5
- 2 × Tesla T4, compute capability 7.5 / SM75
- staged native imports with Kaggle's system PyTorch preserved
- single-T4 inference, raw NCCL all-reduce, and real vLLM TP=2 inference
- Qwen/Qwen2.5-3B-Instruct FP16 TP=2 inference
- persistent TP-aware `sharded_state` save and fresh-engine reload
- OpenAI-compatible `/v1/models` and `/v1/chat/completions` HTTP 200

## Known limitations

- Binary compatibility was validated only for the tabled Kaggle/CPython 3.12
  environment.
- FlashAttention 2 is unavailable on SM75; vLLM selected `TRITON_ATTN`.
- SymmMem communicator warnings are expected on SM75; ordinary NCCL worked.
- FP16, eager execution, and disabled custom all-reduce were conservative
  validation settings, not universal performance recommendations.
- Persistent checkpoints are TP-topology-aware; the Qwen checkpoint was
  validated at TP=2 only.
- The Qwen model artifact is distributed separately under its Qwen Research
  License and is not attached to this GitHub release.
