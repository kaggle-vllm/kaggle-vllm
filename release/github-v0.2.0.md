# kaggle-vllm 0.2.0

`kaggle-vllm` 0.2.0 hardens the small compatibility and runtime-delivery SDK
for the exact validated Kaggle dual NVIDIA Tesla T4 profile. It does not change
or rebuild the immutable native CUDA runtime.

## Highlights

- Add a dependency-aware doctor with human-readable and machine-readable
  `pass`, `warning`, `error`, and `untested` findings.
- Preserve immutable runtime delivery: exact Hugging Face revision, native
  wheel filename and SHA256 verification before `pip --target --no-deps`
  staging, plus a separate dependency overlay.
- Correct the T4/SM75 profile so FlashInfer is optional for the validated
  `TRITON_ATTN` path rather than a false core compatibility failure.
- Harden persistent `sharded_state` inspection against symlink traversal,
  non-contiguous ranks/parts and tensor-parallel topology mismatch.
- Test the lightweight CPU SDK on Python 3.10–3.13 and add package,
  provenance, and documentation-link checks.
- Record real 2026-08-30 Kaggle dual-T4 development-candidate acceptance:
  strict bootstrap and SHA256 verification, preserved system PyTorch, strict
  doctor, native imports, raw NCCL, OPT-125M TP=1/TP=2, local
  OpenAI-compatible HTTP responses, and clean worker termination.
- Record a controlled five-configuration OPT-125M benchmark. For this tiny
  workload TP=1 was faster than TP=2 because communication overhead dominated;
  TP=2 is not claimed as a universal performance improvement.

## Unchanged native runtime

- Upstream project: `vllm-project/vllm`
- Source release: `v0.18.1`
- Source commit: `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`
- Generated wheel distribution version:
  `0.18.2.dev0+ga26e8dc7f.d20260822.cu128`
- Native wheel:
  `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`
- SHA256:
  `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`
- Immutable Hugging Face revision:
  `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`

The generated distribution version comes from `setuptools_scm`; it does not
mean the source was upstream vLLM v0.18.2. Kaggle's preinstalled PyTorch and
CUDA stack remains external and is not replaced by SDK dependency resolution.

## Compatibility boundary

The native CUDA runtime is validated only for Linux x86_64, CPython 3.12.13 /
cp312, PyTorch 2.10.0+cu128, CUDA toolkit 12.8.93, NCCL 2.27.5, and two NVIDIA
Tesla T4 GPUs at compute capability 7.5 / SM75. `TRITON_ATTN` is the expected
attention backend. FlashAttention 2 is unavailable on SM75, and SymmMem is not
a required optimization for ordinary NCCL tensor parallelism.

This project is not an official vLLM distribution, not a Kaggle product, and
not a production-readiness certification. It does not claim general Kaggle GPU
support, arbitrary NVIDIA GPU support, multi-node support, or training support.

## Persistent Qwen artifact

The related `waqasm86/kaggle-vllm-models` repository is a vLLM-native,
rank-specific TP=2 `sharded_state` representation of Qwen2.5-3B-Instruct. It is
not a new, trained, fine-tuned, normal Transformers, or topology-independent
checkpoint. Its Qwen Research License and topology restrictions remain
separate from this SDK's Apache-2.0 license.

## Public SDK artifacts

PyPI published `kaggle-vllm==0.2.0` on 2026-08-31 through GitHub OIDC trusted
publishing from source commit
`020fca67ff197980886c3e725c5c60a6e1478c7c`. Install the lightweight SDK with:

```bash
python -m pip install "kaggle-vllm[hub]==0.2.0"
```

- `kaggle_vllm-0.2.0-py3-none-any.whl` (36,732 bytes), SHA256
  `f3dce393c9e0bd43b9ba29a29ae14f9467857e5eea61390d41f512a52911fbbe`
- `kaggle_vllm-0.2.0.tar.gz` (43,104 bytes), SHA256
  `48ed97da07e54119939053e38f4e87900cf79316711d7b4558aed8122a66e3aa`

These small SDK artifacts contain neither the native CUDA runtime nor Qwen
model files. Those remain authoritative in their separate Hugging Face
repositories.

## Evidence boundary

The accepted development SDK source commit is
`7327b0b0c811a92a9c49421a4d302c18e251ab61`. The separate controlled benchmark
source commit is `6d10912ad73e81f5a62fcec299c87ed5b2631b4f`. The package is now public, but
final acceptance against that exact published identity must still be recorded
before this draft is used to create the immutable `v0.2.0` tag and GitHub
release.
