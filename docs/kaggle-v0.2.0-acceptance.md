# kaggle-vllm 0.2.0.dev0 Kaggle dual-T4 acceptance

Date: 2026-08-30

Status: **PASS**

This report records fresh Kaggle dual-T4 acceptance evidence for the
`kaggle-vllm==0.2.0.dev0` development candidate.

The tested source state was:

- Git repository: `kaggle-vllm/kaggle-vllm`
- Source commit: `6d10912ad73e81f5a62fcec299c87ed5b2631b4f`
- SDK version: `0.2.0.dev0`

This is development-candidate evidence. It does not claim that a final
`kaggle-vllm==0.2.0` PyPI or GitHub release already exists.

## Environment

The fresh Kaggle session validated:

- Python: 3.12.13
- PyTorch: 2.10.0+cu128
- PyTorch CUDA: 12.8
- GPUs: 2 × NVIDIA Tesla T4
- Compute capability: 7.5 / SM75
- NCCL: 2.27.5

## Immutable native runtime

The development SDK continued to use the existing validated native vLLM
artifact:

- Hugging Face repository: `waqasm86/kaggle-vllm-binaries`
- Immutable revision: `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`
- Native wheel:
  `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`
- SHA256:
  `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

No native vLLM rebuild was required for this development-candidate acceptance.

## Acceptance result

The executed notebook completed with:

`FINAL ACCEPTANCE: PASS`

The run validated:

- fresh SDK delivery from the reviewed source state
- strict native-runtime bootstrap
- native imports for `vllm`, `vllm._C`, `vllm._moe_C`, and
  `vllm.cumem_allocator`
- preservation of Kaggle's system PyTorch installation
- successful strict `kaggle-vllm doctor`
- raw dual-rank NCCL communication
- real OPT-125M TP=1 generation
- real OPT-125M TP=2 generation
- local OpenAI-compatible serving
- HTTP 200 responses from the tested endpoint requests
- clean worker termination

The Qwen TP=2 sharded-state regression was not repeated in this focused run.
The native runtime and previously validated Qwen artifact were unchanged.

## Tesla T4 behavior

Tesla T4 is SM75, so FlashAttention 2 is not available on this GPU generation.
The validated vLLM path uses `TRITON_ATTN`.

SymmMem capability warnings are also expected on SM75 and did not prevent
ordinary NCCL tensor-parallel execution.

## Evidence

Machine-readable acceptance evidence and the OpenAI server log are retained in:

`artifacts/kaggle-2026-08-30-v0.2.0-acceptance/`

The executed notebook is retained as:

`kaggle-notebooks/kaggle_vllm_0_2_0_acceptance.ipynb`

The evidence should be interpreted as validation of the exact development
candidate identified above, before promotion to a final 0.2.0 release.
