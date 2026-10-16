# Gemma 4 offline four-L4 experiments

These notebooks run in the authorized Gemma 4 Developer Agent competition with Internet disabled and a four-NVIDIA-L4 accelerator. Attach these existing Kaggle Inputs to each GPU notebook:

| Input | Contents |
| --- | --- |
| `waqasm86/kaggle-vllm-v030-py312-runtime-core-v2` | P33 core dependency wheels |
| `waqasm86/kaggle-vllm-v030-py312-runtime-large-v2` | P33 large dependency wheels |
| `waqasm86/kaggle-vllm-v030-py312-torch-cu129-v2` | P33 PyTorch/CUDA dependency wheels |
| `waqasm86/kaggle-vllm-v030-py312-wheelhouse-audit-v2` | Prior source audit, not an installation source |
| `waqasm86/kaggle-vllm-v030-py312-build-deps-v2` | Pinned Jinja2 and Packaging wheels needed by the current separate Python 3.12 interpreter |
| `waqasm86/kaggle-vllm-v031-py312-base-private` | Pinned OpenAI 2.44.0 and Distro 1.9.0 wheels needed by vLLM's lazy `LLM` import in that interpreter |
| `waqasm86/vllm-0-30-0-cp312-cp312-linux-x86-64` | Frozen native wheel |
| `waqasm86/kaggle-vllm-sdk-v031-l4-acceptance` | Published SDK wheel, acceptance scripts, read-only pip bootstrap, TP4 trial |
| Kaggle Model `google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2` | Licensed model weights and metadata |

The first three wheelhouse Inputs jointly contain exactly the 168 filenames in the published SDK's `L4_INFERENCE_LOCK.json`. The first fresh four-L4 audit verified those 168 hashes and the exact SDK/native/model bytes. Rerun `01_offline_inputs_audit.ipynb` with the new supplemental base Input attached to hash its OpenAI and Distro wheels as well. Do not install or claim a working SDK if it fails. The 0.3.1 SDK wheel must hash to `a14a0be361ea0ab876aa939743ad7259ac012b8c9410cc668899da00b875c27b`; the native wheel must hash to `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`.

`02_sdk_native_runtime_offline.ipynb` runs the 19-gate acceptance script in `/usr/bin/python3.12`, because the current Kaggle notebook kernel is Python 3.13. The attached `pip-25.2-py3-none-any.whl` is placed on `PYTHONPATH` as a read-only zip. It is not installed into Kaggle's system Python. This separate Python 3.12 interpreter needs pinned Jinja2 3.1.6, Packaging 26.3, MarkupSafe 3.0.4, OpenAI 2.44.0, and Distro 1.9.0; the notebook verifies their wheel hashes before installing them to an isolated `/tmp` target. The SDK then stages the complete locked runtime under `/tmp/kaggle-vllm/v0.3.0/l4x4`, verifies hashes before installation, and uses `--target --no-index --no-deps`. The four-L4 session never downloads from Hugging Face CDN. A runner exit of zero still needs its `19_final_matrix.json`, per-gate evidence, and manifest hashes reviewed before acceptance is declared.

`03_gemma4_tp4_inference.ipynb` measures separate local SDK inference. `04_tp4_sharded_state_research.ipynb` tests vLLM's supported four-rank save API and reload on a fresh session with enough working storage; it refuses to call a checkpoint validated until rank completeness, hashes, deterministic readback, and cleanup pass. `05_competition_submission_paper.ipynb` checks local submission packaging and evidence pointers. Each GPU notebook expects a fresh session; do not run the installers successively in one kernel. Keep all runtime Inputs and the model read-only. Do not attach the runtime wheels to a public output dataset without a separate third-party redistribution notice review.

`06_official_agent_task_trial.ipynb` is a separate agent-quality experiment. Attach the official competition Input, `metric/gemma-4-developer-agent-wheelhouse`, the same 31B Gemma Model Input, the private `waqasm86/gemma4-agent-v031-private-input` Dataset, the small SDK Input for its read-only pip wheel, and private `waqasm86/gemma4-adk-litellm-18314-private` for the Google ADK bridge. The Kaggle notebook kernel may be Python 3.13; this notebook verifies the 54-wheel private bridge manifest and stages it separately from the organizer's 41-wheel closure under `/tmp` with `/usr/bin/python3.12`, then runs two public training issues through the official ADK harness. Inspect its patch and test logs before reporting task outcomes. The `kaggle-vllm` inference notebooks do not install LiteLLM.

The submission ZIP under `competition/submission.zip` is a Google ADK harness package and uses the organizer's scorer. Local `kaggle-vllm` inference is an experiment, not a replacement serving interface for the scored agent.
