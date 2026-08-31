# kaggle-vllm 0.2.0 final Kaggle acceptance

The exact public `kaggle-vllm==0.2.0` package and the existing Qwen TP=2
`sharded_state` artifact passed their final release gates on a fresh Kaggle
session with two NVIDIA Tesla T4 GPUs on 2026-08-31.

## Distinct tested identities

These evidence lines are related but are not interchangeable:

| Evidence | SDK/source identity | Date | Result |
|---|---|---|---|
| Development-candidate acceptance | `0.2.0.dev0`, source `7327b0b0c811a92a9c49421a4d302c18e251ab61` | 2026-08-30 | PASS |
| Controlled benchmark | source `6d10912ad73e81f5a62fcec299c87ed5b2631b4f` | 2026-08-30 | PASS |
| Public PyPI package | `kaggle-vllm==0.2.0`, published from `020fca67ff197980886c3e725c5c60a6e1478c7c` | 2026-08-31 | Published and locally verified |
| Final post-publication acceptance | exact public `kaggle-vllm==0.2.0` | 2026-08-31 | PASS |
| Final Qwen TP=2 regression | exact public SDK plus immutable model revision | 2026-08-31 | PASS |

The unchanged native runtime comes from upstream vLLM v0.18.1 at
`a26e8dc7ff2111a005144d775ecf9cebf56c45b2`. Its generated distribution
version is `0.18.2.dev0+ga26e8dc7f.d20260822.cu128`; that generated metadata
does not mean the source was upstream v0.18.2.

## Final public-package acceptance

The executed
[`kaggle_vllm_0_2_0_post_publication_acceptance.ipynb`](../kaggle-notebooks/kaggle_vllm_0_2_0_post_publication_acceptance.ipynb)
installed the exact public `kaggle-vllm[hub]==0.2.0` package. It recorded:

| Component | Observed identity |
|---|---|
| Platform | Kaggle Notebook, Linux/glibc 2.35 |
| Python | CPython 3.12.13 |
| GPUs | 2 × NVIDIA Tesla T4, SM75 |
| PyTorch before/after | 2.10.0+cu128 at the unchanged system path |
| PyTorch CUDA ABI | 12.8 |
| NCCL | 2.27.5 |
| Native wheel | `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl` |
| Native SHA256 | `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c` |
| Native HF revision | `f6b4f10de54924ed6fe9e28cceab84eca7276ab6` |

Every final check was true:

- environment identity;
- strict bootstrap and immutable SHA256 verification;
- preservation of Kaggle's system PyTorch;
- imports of `vllm`, `vllm._C`, `vllm._moe_C`, and
  `vllm.cumem_allocator`;
- strict dependency-aware doctor;
- raw two-rank NCCL;
- OPT-125M TP=1 and TP=2 generation;
- OpenAI-compatible `/v1/models` HTTP 200;
- OpenAI-compatible `/v1/completions` HTTP 200;
- clean server process-group shutdown.

The notebook is nbformat 4.5 with 15 cells and eight code cells. Seven code
cells carry execution counts 1–7; the only unexecuted code cell is an empty
trailing cell. No Jupyter exception output is present. The final marker is
`FINAL PUBLISHED 0.2.0 ACCEPTANCE: PASS`.

The original JSON sidecar, OpenAI server log, and runtime manifest were not
downloaded separately. The notebook printed the complete final JSON object,
which was recovered field-for-field into
[`kaggle-vllm-020-published-acceptance-evidence.recovered.json`](../artifacts/kaggle-2026-08-31-v0.2.0-final-acceptance/kaggle-vllm-020-published-acceptance-evidence.recovered.json).
It is explicitly identified as recovered notebook-output evidence rather than
an untouched raw Kaggle file.

## Final Qwen TP=2 regression

The executed
[`kaggle_vllm_0_2_0_qwen_regression.ipynb`](../kaggle-notebooks/kaggle_vllm_0_2_0_qwen_regression.ipynb)
used the existing immutable `waqasm86/kaggle-vllm-models` artifact at revision
`08bb62d0b68d20062e9009a9769c0df53d3dae21`.

| Property | Recorded result |
|---|---|
| Inspection mode | `direct-real-artifact` |
| Tensor-parallel topology | TP=2, ranks 0–1 |
| Rank/part layout | two ranks × two parts |
| Total sharded bytes | 6,172,262,512 |
| Symlink state | all four shard files are regular files, not symlinks |
| Missing metadata | none |
| Topology errors | none |
| Inspection validity | true |

The exact shards were:

| File | Bytes |
|---|---:|
| `model-rank-0-part-0.safetensors` | 2,138,586,872 |
| `model-rank-0-part-1.safetensors` | 947,544,384 |
| `model-rank-1-part-0.safetensors` | 2,138,586,872 |
| `model-rank-1-part-1.safetensors` | 947,544,384 |

The structural inspection, TP=2 identity, topology-mismatch rejection, symlink
policy, real TP=2 `sharded_state` load, short generation, and clean child
process exit all passed. The sentinel identity is
`KAGGLE-VLLM-QWEN-REGRESSION-V3` with status `PASS`.

The copied Hugging Face `model.safetensors.index.json` references ordinary HF
weight filenames that are not present in this topology-specific directory.
That informational warning is expected: vLLM's `sharded_state` loader uses the
rank-specific files listed above.

## Runtime-log interpretation

The preserved Qwen log records `load_format=sharded_state`, TP=2, FP16,
`max_model_len=2048`, eager execution, and disabled custom all-reduce. Two
workers initialized as ranks 0 and 1 with NCCL 2.27.5. SymmMem was unavailable
on compute capability 7.5 and FlashAttention 2 reported its expected SM80+
requirement; vLLM selected `TRITON_ATTN` on the T4 profile.

The model loaded through `sharded_state_loader.py`, used approximately 2.93 GiB
of memory, and emitted real generated text. `EngineCore` then logged `Shutdown
initiated` followed by `Shutdown complete`; both TP workers recorded parent exit
and worker shutdown. A subsequent client-side line reports that the already
terminated EngineCore died unexpectedly. That teardown-side client message is
retained verbatim in the log. It is not an inference or checkpoint failure:
generation had succeeded, shutdown had completed, and the explicit
`clean_child_process_exit` assertion passed.

## Evidence inventory

Small evidence and exact checksums are under
[`artifacts/kaggle-2026-08-31-v0.2.0-final-acceptance/`](../artifacts/kaggle-2026-08-31-v0.2.0-final-acceptance/).
No Qwen weights, native CUDA wheel, Torch package, CUDA library, or cache is
committed to Git.
