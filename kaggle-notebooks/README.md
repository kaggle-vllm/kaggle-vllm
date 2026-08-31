# Kaggle notebooks

This directory contains Kaggle-specific validation material for
`kaggle-vllm`. An executed acceptance notebook is release evidence; templates
under [`examples/`](../examples/) are starting points and do not claim a
successful run until their outputs have been captured and reviewed.

## Post-0.2.0 Milestone 1 evidence

[`kaggle_vllm_milestone_1_tp_diagnostics.ipynb`](kaggle_vllm_milestone_1_tp_diagnostics.ipynb)
is the executed Milestone 1 dual-T4 TP performance diagnostic notebook. It ran
the reviewed source identity
`7e7355d64266e864a8113c30d52c612d98100350` on a real Kaggle T4 x2 session,
and all six planned OPT-125M and Qwen configurations completed. The notebook
retains its real outputs; checksummed JSON, logs and topology evidence are in
[`artifacts/kaggle-2026-09-01-milestone-1/`](../artifacts/kaggle-2026-09-01-milestone-1/).
See the [benchmark and interpretation guide](../docs/benchmarking.md).

## v0.2.0 development-candidate Kaggle evidence

The 2026-08-30 fresh-session Kaggle validation for `0.2.0.dev0` is complete.

[`kaggle_vllm_0_2_0_acceptance.ipynb`](kaggle_vllm_0_2_0_acceptance.ipynb)
is the executed dual-T4 acceptance notebook. It ends in
`FINAL ACCEPTANCE: PASS`.

[`kaggle_vllm_0_2_0_benchmark.ipynb`](kaggle_vllm_0_2_0_benchmark.ipynb)
is the executed controlled TP=1/TP=2 benchmark notebook. All five benchmark
configurations were attempted and completed successfully.

The acceptance tested source commit
`7327b0b0c811a92a9c49421a4d302c18e251ab61`; the controlled benchmark tested
the later commit `6d10912ad73e81f5a62fcec299c87ed5b2631b4f`. Both used SDK version
`0.2.0.dev0`. The separate identities are retained in their machine-readable
records.

The environment remained the validated Kaggle configuration:

- Python 3.12.13
- PyTorch 2.10.0+cu128
- CUDA 12.8
- 2 × Tesla T4 / SM75
- NCCL 2.27.5
- vLLM attention backend: `TRITON_ATTN`

The underlying native vLLM wheel remained the existing immutable Hugging Face
artifact; no native CUDA rebuild was required.

Human-readable reports are available in:

- [`docs/kaggle-v0.2.0-acceptance.md`](../docs/kaggle-v0.2.0-acceptance.md)
- [`docs/kaggle-v0.2.0-benchmark.md`](../docs/kaggle-v0.2.0-benchmark.md)

Machine-readable evidence is retained under:

- [`artifacts/kaggle-2026-08-30-v0.2.0-acceptance/`](../artifacts/kaggle-2026-08-30-v0.2.0-acceptance/)
- [`artifacts/kaggle-2026-08-30-v0.2.0-benchmark/`](../artifacts/kaggle-2026-08-30-v0.2.0-benchmark/)

These results validate the development candidate and remain distinct from the
final public-package evidence below.

## v0.2.0 final release gates

[`kaggle_vllm_0_2_0_post_publication_acceptance.ipynb`](kaggle_vllm_0_2_0_post_publication_acceptance.ipynb)
is the executed fresh dual-T4 run after public PyPI publication. It installed
exactly `kaggle-vllm[hub]==0.2.0` and ends in
`FINAL PUBLISHED 0.2.0 ACCEPTANCE: PASS`.

[`kaggle_vllm_0_2_0_qwen_regression.ipynb`](kaggle_vllm_0_2_0_qwen_regression.ipynb)
is the executed V3 diagnostic regression for the already-published Qwen TP=2
`sharded_state` artifact. It used the existing real artifact without
regenerating model shards and ends in `FINAL QWEN TP=2 REGRESSION: PASS`.

The reviewed [final acceptance report](../docs/kaggle-v0.2.0-final-acceptance.md)
and [small evidence directory](../artifacts/kaggle-2026-08-31-v0.2.0-final-acceptance/)
record exact identities, recovery provenance, checksums, and runtime-log
interpretation.

## v0.1.2 benchmark evidence

[`kaggle_vllm_0_1_2_benchmark.ipynb`](kaggle_vllm_0_1_2_benchmark.ipynb)
is the executed 2026-08-30 fresh-session TP=1/TP=2 benchmark for the published
`kaggle-vllm==0.1.2` SDK.

All five controlled OPT-125M configurations executed successfully. The
corresponding [benchmark report](../docs/kaggle-v0.1.2-benchmark.md) separates
functional acceptance from performance conclusions. For this small model,
TP=1 outperformed TP=2 and non-eager execution substantially outperformed the
conservative eager acceptance configuration.

Machine-readable results and raw logs are retained under
[`artifacts/kaggle-2026-08-30-v0.1.2-benchmark/`](../artifacts/kaggle-2026-08-30-v0.1.2-benchmark/).

## v0.1.2 focused reset acceptance evidence

[`kaggle_vllm_0_1_2_reset_acceptance.ipynb`](kaggle_vllm_0_1_2_reset_acceptance.ipynb)
is the executed 2026-08-25 focused acceptance run for the safe runtime-reset
feature added in v0.1.2. Its saved outputs end in `FINAL ACCEPTANCE: PASS`; the
corresponding [evidence report](../docs/kaggle-v0.1.2-reset-acceptance.md)
records the independent review.

The focused notebook verified:

- public PyPI `kaggle-vllm==0.1.2` on Python 3.12.13
- 2 × Tesla T4 at SM75 with PyTorch 2.10.0+cu128 and NCCL 2.27.5
- initial strict bootstrap from the unchanged immutable native wheel
- the expected refusal to overwrite an unowned non-empty staged directory
- a non-mutating manifest-aware reset dry-run
- explicit reset of staged/overlay/manifest state with cache preservation
- immediate strict re-bootstrap and staged native-extension imports
- unchanged Kaggle system Torch and real OPT-125M NCCL TP=2 generation

It intentionally does not repeat the Qwen download or reload. The v0.1.2
package change is limited to runtime-reset safety; the native wheel and Qwen
TP=2 model are unchanged, and the v0.1.1 notebook remains the full delivery
and Qwen acceptance record.

## v0.1.1 acceptance evidence

[`kaggle_vllm_0_1_1_dual_t4_acceptance.ipynb`](kaggle_vllm_0_1_1_dual_t4_acceptance.ipynb)
is the executed 2026-08-25 post-PyPI, post-Hugging-Face-rename acceptance run.
Its outputs are intentionally retained. The corresponding human-readable
summary is in [the acceptance report](../docs/kaggle-v0.1.1-acceptance.md).

The notebook validated this environment:

| Component | Observed value |
|---|---|
| Kaggle runtime | Linux, glibc 2.35 |
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA | 12.8 |
| CUDA toolkit | 12.8.93 |
| GPUs | 2 × NVIDIA Tesla T4 |
| Compute capability | 7.5 / SM75 |
| NCCL | 2.27.5 |

To reproduce the hardware setup, enable **Internet** and select the Kaggle
accelerator **GPU T4 x2** before starting the session. A Hugging Face token is
optional for public artifacts but can help with Hub rate limits. If used, add
it as a Kaggle Secret named `HF_TOKEN`; retrieve it through Kaggle's secrets
API and report only whether it was found. Never print or save the token in a
notebook output.

## Runtime layout and storage

The acceptance run deliberately used notebook-owned paths:

```text
/kaggle/working/kaggle-vllm-e2e-011/vllm-staged
/kaggle/working/kaggle-vllm-e2e-011/vllm-runtime-overlay
/kaggle/working/kaggle-vllm-e2e-011/kaggle-vllm-runtime.json
/kaggle/working/kaggle-vllm-cache
```

Using a new isolated root preserves the SDK's refusal to overwrite non-empty
runtime destinations and avoids mixing files from earlier attempts. The cache
can remain reusable across compatible runs.

The Kaggle UI showed approximately 2.2 GiB for the native bootstrap/runtime
and about 8 GiB of total working/output usage after downloading and loading the
Qwen TP=2 checkpoint. The four rank-specific Qwen weight files total exactly
6,172,262,512 bytes (about 5.75 GiB). These are observed reference values, not
universal guarantees; cache state and filesystem accounting affect totals.

## Expected Tesla T4 messages

Tesla T4 is SM75. FlashAttention 2 requires a newer compute capability, so its
unavailability message is expected and vLLM's validated selection is
`TRITON_ATTN`. SymmMem communicator capability warnings are also expected on
SM75. Neither prevented ordinary NCCL TP=2 communication or generation in the
acceptance run.
