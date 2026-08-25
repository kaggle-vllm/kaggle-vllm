# Kaggle notebooks

This directory contains Kaggle-specific validation material for
`kaggle-vllm`. An executed acceptance notebook is release evidence; templates
under [`examples/`](../examples/) are starting points and do not claim a
successful run until their outputs have been captured and reviewed.

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
