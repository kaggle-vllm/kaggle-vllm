# kaggle-vllm v0.1.2

`kaggle-vllm` 0.1.2 adds an explicit, manifest-aware workflow for safely
resetting SDK-owned native runtime state. The normal refusal to overwrite a
non-empty destination remains the default behavior.

## Safe runtime reset

- inspect the exact plan without mutation using
  `kaggle-vllm bootstrap --reset-runtime --dry-run`
- require explicit `--yes` before destructive reset
- remove only validated staged, overlay, and manifest resources
- preserve the checksum-verified download cache by default
- reject dangerous, overlapping, symlinked, or unowned custom paths
- immediately continue strict bootstrap after a confirmed reset

See [issue #7](https://github.com/kaggle-vllm/kaggle-vllm/issues/7) and
[PR #9](https://github.com/kaggle-vllm/kaggle-vllm/pull/9).

## Focused Kaggle acceptance

The executed
[v0.1.2 reset acceptance notebook](https://github.com/kaggle-vllm/kaggle-vllm/blob/v0.1.2/kaggle-notebooks/kaggle_vllm_0_1_2_reset_acceptance.ipynb)
and
[evidence report](https://github.com/kaggle-vllm/kaggle-vllm/blob/v0.1.2/docs/kaggle-v0.1.2-reset-acceptance.md)
record `FINAL ACCEPTANCE: PASS` on 2026-08-25:

- Python 3.12.13 and preserved PyTorch 2.10.0+cu128 / CUDA 12.8
- 2 × Tesla T4 at SM75 with NCCL 2.27.5
- initial strict bootstrap from the canonical immutable Hugging Face artifact
- expected default refusal against an unowned non-empty staged destination
- non-mutating reset dry-run
- explicit manifest-owned reset with cache preservation
- successful strict re-bootstrap and native extension imports
- real OPT-125M NCCL TP=2 initialization and generation
- expected SM75 selection of `TRITON_ATTN` and non-fatal SymmMem warnings

## Unchanged native and model artifacts

The native vLLM artifact was not rebuilt:

- repository: `waqasm86/kaggle-vllm-binaries`
- immutable revision: `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`
- wheel: `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`
- SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

The Qwen TP=2 `sharded_state` model is also unchanged. This focused reset
acceptance did not redownload or revalidate Qwen; the full v0.1.1 acceptance
remains its current evidence.

## Distribution

- [PyPI SDK](https://pypi.org/project/kaggle-vllm/)
- [GitHub source](https://github.com/kaggle-vllm/kaggle-vllm)
- [Native vLLM binaries](https://huggingface.co/waqasm86/kaggle-vllm-binaries)
- [Qwen TP=2 persistent model](https://huggingface.co/waqasm86/kaggle-vllm-models)

GitHub release assets contain only the exact small PyPI SDK wheel, source
distribution, and checksum manifest. Large native CUDA and Qwen model artifacts
remain authoritative on Hugging Face and are intentionally not duplicated.
