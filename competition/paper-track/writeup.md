# Does four-L4 tensor parallelism help an offline Gemma 4 coding agent finish repairs within a fixed time budget?

**Draft status, 16 October 2026:** Original research design with verified historical runtime observations. Fresh SDK 0.3.1 results, agent-task outcomes, and ablations must be filled from executed evidence before this is submitted. This draft is under the Paper Track's 3,000-word limit.

## Abstract

An offline coding agent must turn inference capacity into correct repository patches under a fixed wall-clock budget. We study whether four-way tensor parallelism (TP4) for the quantized Gemma 4 31B instruction model on four NVIDIA L4 GPUs improves that outcome, and which parts of the end-to-end budget are spent on runtime initialization, generation, tool use, and tests. The system uses an exact CUDA 12.9/PyTorch 2.13.0+cu129/vLLM 0.30.0 runtime delivered by the Python `kaggle-vllm` 0.3.1 SDK for controlled local inference. The scored agent uses the organizer's Google ADK harness interface. We predefine a task-level pass/fail metric and paired ablations so that generation speed is not mistaken for code-repair quality. Historical 0.3.0 evidence establishes TP4 feasibility, but does not answer the agent question; the new measurements remain pending.

## Research question and prior work

**Question:** On a fixed four-L4 allocation and 12-hour aggregate evaluation budget, how much of any change in code-repair success is explained by TP4 inference and initialization latency, compared with agent workflow choices such as code search, editing, and verification?

Tensor parallelism partitions model computation and storage across devices [1]. vLLM provides the serving engine and memory management [2]. Issue-level patch evaluation, as in SWE-bench [3], measures a different outcome from tokens per second. The organizer's [Gemma 4 Developer Agent harness][4] applies submitted patches and runs validation tests; its submission ZIP is declarative ADK configuration, not a replacement inference server. Our earlier public SDK documentation, engineering article, and **published Zenodo dual-T4 crossover paper** [6] analyze runtime compatibility and TP performance on smaller models. They are prior work, not evidence of Gemma 4 agent improvement. This study changes the hardware, model, outcome, and experimental unit to issue-level patch success; it must add new Gemma 4 issue-level evidence rather than reuse the earlier paper's result as its contribution.

## System and reproducibility

The fixed local inference target is Linux x86_64, CPython 3.12, four NVIDIA L4 GPUs (SM89), CUDA runtime 12.9, PyTorch 2.13.0+cu129, upstream vLLM 0.30.0, and `kaggle-vllm` SDK 0.3.1. The SDK wheel SHA256 is `a14a0be361ea0ab876aa939743ad7259ac012b8c9410cc668899da00b875c27b`; the frozen native wheel SHA256 is `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`. The P33 offline inference closure has 168 dependency wheels, each verified against its lock before installation. Gemma 4 `gemma-4-31b-it-qat-w4a16-ct` is mounted as a licensed Kaggle Model Input. L4 sessions have Internet disabled. The Python 3.12 runtime is staged under `/tmp`; Kaggle's preinstalled system packages are not replaced. We record the exact mounted model version, model config hash, driver, CUDA and NCCL versions, import origins, attention backend per rank, input manifest hashes, and process cleanup.

The organizer's current `HARNESS_README.md` (13 October 2026) permits four quantized Gemma 4 variants and requires the same declared base model across all agents. This study fixes the 31B QAT model throughout. The organizer's scorer controls its own vLLM server and ADK binding. The local SDK experiment and the scored submission therefore have separately recorded runtime identities.

## Experimental design

We will freeze a subset of public training issues before looking at results, stratified by repository and task type, and record each task ID and base commit. Each trial uses the same model, model revision, prompt budget, task snapshot, and five-minute per-task cap. The unit of analysis is one task/run seed, not an individual generated token or tool call. A patch passes only when the organizer's official verification tests report all required tests passing. We also record whether a patch was generated, whether the agent ran relevant tests, wall time, tool calls, failure type, and time spent in model inference versus tools. We will run more than one seed and report both per-task paired results and aggregate uncertainty intervals. No task will be counted as success from a plausible-looking diff alone.

The primary agent is a single ADK `LlmAgent` using permitted repository, graph-search, command, edit, status, and patch-submission tools. The submitted YAML declares the 31B model; `adk-submission` 0.2.13 validates its directory and compiles it to an ADK agent. The official harness, rather than `kaggle-vllm`, serves this scored agent. For the local inference track, `KaggleLLM.chat` runs Gemma at TP4 with deterministic sampling, `max_model_len=2048`, and the accepted eager/collective compatibility settings. We record initialization time, request latency, generated tokens, sampled GPU memory/utilization, backend, and worker cleanup. Request tokens divided by total request time are labelled as such, not steady-state decode throughput.

**Baselines and ablations, all pending execution:** (A) organizer sample ADK agent versus our single-agent workflow on identical issues; (B) graph search enabled versus disabled, same prompt and model; (C) the five-minute verification budget versus a shorter cap; (D) local TP4 eager versus graph mode when supported on the exact runtime, measuring inference only; (E) TP1 versus TP4 only if the same 31B model fits one L4 under a documented identical context and memory guard. Failed capacity or compatibility gates are reported as missing performance, not zero throughput. We will not extrapolate local TP timing directly to the scorer's different serving stack.

## Verified observations and open measurements

| Observation | Evidence | Interpretation |
| --- | --- | --- |
| Historical 0.3.0 L4 acceptance: 19/19 mandatory gates passed; Gemma 4 generated 40 tokens using TP4 and `TRITON_ATTN` on all four ranks; cleanup passed | `docs/release/v0.3.0-acceptance-audit.md` | Native wheel feasibility only; not fresh SDK 0.3.1 evidence |
| P33 native qualification: engine initialization 338.807 s, one semantic generation 5.413 s, 33 output token IDs | `p33-kaggle-vllm-v030-l4x4-production-qualified-evidence.zip`, `evidence/03_gemma_tp4_result.json` | Single eager-mode request; no throughput distribution or agent score |
| Published SDK 0.3.1 wheel and unchanged native wheel | PyPI/GitHub/Hugging Face exact SHA256 API checks | Artifact availability; not an agent benchmark |
| Fresh 0.3.1 TP4 acceptance | Pending `02_sdk_native_runtime_offline.ipynb` output | Must pass 19/19 and evidence integrity |
| Paired issue resolution and ablations | Pending official harness runs | Required for the research question |

## Limitations and integrity

Four L4 GPUs share a single constrained Kaggle environment, so results will not establish behavior on other interconnects or accelerators. Eager execution and a 2,048-token local context limit differ from the organizer scorer's 32,768-token server configuration. GPU utilization polling can miss short peaks. A single semantic prompt tests neither patch quality nor general throughput. Existing 0.3.0 acceptance cannot substitute for 0.3.1 integration. We will disclose unsuccessful runs, allocation failures, and any task excluded before evaluation. The previously **published** dual-T4 paper [6] has overlapping SDK, provenance, and benchmarking methods; its numerical results cannot be presented as new findings here. This paper is not submission-ready without newly executed Gemma 4 agent-task data and analysis.

## References

1. Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), 2019.
2. Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023.
3. Jimenez et al., [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770), 2023.
4. Google DeepMind and Kaggle, [Gemma 4 Developer Agent Competition overview](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview), 2026.
5. Google DeepMind and Kaggle, [Gemma 4 Developer Agent Paper Track submission requirements](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/overview/submission-requirements), 2026.
6. Waqas, [Characterizing Tensor-Parallel Crossover in vLLM Serving on Dual NVIDIA Tesla T4 GPUs](https://doi.org/10.5281/zenodo.23119478), Zenodo, 2026. Published prior work by the same author.

[4]: https://www.kaggle.com/competitions/gemma-4-developer-agent/overview
