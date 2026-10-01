# Supplementary bilingual serving case study: ALLaM-7B

Status: **REVIEWED_SUPPLEMENTARY_EVIDENCE**. This post-M4 case study uses a
different protocol. ALLaM is not a fifth principal M4 model or another M4
repetition. It changes neither the four-model, 60-shard population (55 canonical
and five terminal resource outcomes; 720 planned cells) nor its measurements,
uncertainty treatment, figures, crossover conclusions or publication gates.

[Evidence and integrity instructions](../../artifacts/kaggle-2026-10-01-allam-7b/README.md)
link the three source notebooks, byte-preserved executions, runner sources,
request metrics, telemetry and logs. [summary.json](summary.json) and
[results.md](results.md) are generated from machine evidence, which outranks
historical notebook prose.

## Runtime and model identity

All three runtime manifests record Python 3.12.13, PyTorch 2.10.0+cu128,
CUDA 12.8, toolkit 12.8.93, NCCL 2.27.5, driver 580.159.04 and two Tesla T4
SM75 devices. Driver-reported CUDA maximum 13.0 is not the runtime CUDA version.
The SDK is `kaggle-vllm==0.2.0`. The immutable native build derives from upstream
vLLM v0.18.1, commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2` (see the
repository profile for the authoritative full source identity); the generated
distribution is `0.18.2.dev0+ga26e8dc7f.d20260822`.

- Model: `humain-ai/ALLaM-7B-Instruct-preview`
- Revision: `a28dd1e67420cde72d3629c8633a974cf7d9c366`
- Internal version: `7b-alpha-v2.33.0.30`
- Wheel: `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl`
- Wheel SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`
- Binary repository revision: `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`

The BF16 checkpoint is loaded as FP16, with eager execution, custom all-reduce
disabled, spawn workers and TRITON_ATTN on SM75. TP2 rank assignments and NCCL
initialization appear for both physical devices; visibility alone is not used
as proof of tensor parallel execution. Upstream vLLM supplies tensor
parallelism, scheduling, kernels and attention; kaggle-vllm supplies the Kaggle
compatibility/runtime delivery layer.

## P2: compatibility

P2 resolved the model to the recorded immutable commit and constructed TP2 at
context 2048 and utilization 0.85. Initialization took 110.3194 s; the bilingual
smoke generation took 7.7651 s, with 36 prompt and 17 output tokens total.
English and Arabic responses are nonempty. The log resolves `LlamaForCausalLM`,
reports 5.35 GiB available KV memory, 21,904 cache tokens and 10.70x estimated
2048-token concurrency. These are engine diagnostics, not measured serving
throughput or a language-quality score. Final process/GPU output reports no
running GPU processes.

## P3: exploratory offline characterization

P3 uses `KaggleLLM`/`LLM.generate()` in subprocesses, TP1 utilization 0.98,
TP2 utilization 0.85 and `max_num_seqs=8`. It retains six outcomes: TP1 passes
960 and resource-gates 2048/4096; TP2 passes all three contexts. Each successful
configuration contains an excluded warmup, three 128-input/32-output core
repetitions per language, two near-context probes, one offline wave at each
concurrency 1/2/4/8, and four chat probes. Batch throughput divides actual output
tokens by the wall time of `generate()`. Request latency fields are unavailable
(null), not zero.

The offline concurrency sweep uses English at 1 and Arabic at 2/4/8, requests
identical token-ID prompts within a batch, and measures only one wave per point.
Its curves confound language, caching and concurrency and are not used to infer
a language-controlled serving crossover. Small tokenizer diagnostics are
whitespace-word counts over synthetic snippets, not a corpus-level fertility
study or a comparison against another tokenizer.

The notebook is a composite exploration: configuration cell execution count 12
precedes the saved matrix-driver count 13; four prior outcomes are explicitly
reused, and two 960-token configurations are added. No immutable pre-execution
source freeze exists. Embedded runner bytes match the saved runner; raw results,
logs, original manifests and regenerated tables agree. This supports the
retained observations, not a claim that every cell ran once in saved order or
that post-execution editing can be conclusively excluded. Missing result JSON
paths for failed startups are intended output destinations, not lost successful
measurements. Historical narrative saying 4032 input tokens belongs to P3's
context-minus-64 probe and must not be substituted for P4's 4000-input probe.

## P4: final online serving protocol

Each of six configurations starts a separate loopback OpenAI-compatible vLLM
server. Five become ready; TP1/4096 fails the KV-cache capacity check. Each viable
server warms both languages, then runs English and Arabic separately, client
request concurrency 1/2/4/8, and three sequential waves per point. Each wave
launches exactly C asynchronous requests. Concurrency is a client property,
not an observed instantaneous decode-batch size. There are 120 measured waves
and 450 measured requests, all successful with 128 actual prompt tokens and
32 actual output tokens. Unique early request prefixes reduce deliberate
prefix reuse; prefix caching remains enabled and absence of cache hits is not
proved. Deterministic prompts and the client source are retained.

Three waves share one server lifetime per configuration. They are not independent
fresh-server or allocation repetitions, and no confidence interval, universal
crossover or causal mechanism is inferred. Fixed language/configuration order,
short waves and only 3/6/12/24 pooled requests per latency point limit tail
percentile interpretation. TP1 and TP2 also use different memory budgets.

Metric definitions from the saved streaming client:

- TTFT: request start to first **nonempty text chunk** received by the client.
- E2E: request start through stream completion, including HTTP/client overhead.
- TPOT: `(E2E - TTFT) / (completion_tokens - 1)`; a client-derived average,
  not a direct per-token CUDA measurement. Chunk interarrival is not token ITL.
- Wave output throughput: sum of actual completion tokens / whole-wave wall
  seconds. Request throughput uses successful request count / that same wall.
  Total-token throughput includes prompt tokens as well.
- Published throughput: arithmetic mean of three wave rates, not sum of tokens
  divided by sum of wave durations. Sample standard deviations and medians are
  retained. Latency p50/p95/p99 pool successful requests across those waves and
  use linear-interpolated quantiles, not means of wave percentiles.

[Generated results](results.md) show TP1/TP2 close at concurrency 1 at matched
context 960; TP2's observed mean advantage is larger at 8. The capacity benefit
of TP2/4096 is separate from its measured throughput differences. TP1/2048
near-context probes complete 1952+32=1984 tokens in both languages; TP2/4096
completes 4000+32=4032. This does not establish all possible full-context
workloads. Resource-gated performance is N/A, never zero.

GPU telemetry records sampled device memory/utilization including startup and
teardown, not tensor allocation or an unsampled true peak. P4 reports 14,671–14,799 MiB on the active successful TP1 GPU and 12,871–13,001 MiB per
TP2 GPU (exact values in the configuration CSV). The idle TP1 device can
legitimately record zero memory; that is not zero performance. Server logs
show shutdown, and matrix records show return code 0 for viable servers.
TP1 shutdown warns that `destroy_process_group()` was not called. The driver
signals process groups and waits; this is not proof that every process-group
resource was cleanly destroyed. No claim of warning-free teardown is made.

## P3/P4 context discrepancy

The observed single-GPU context boundary was execution/configuration dependent.
The final online serving protocol supported 2048 tokens but resource-gated 4096;
earlier P3 observations therefore must not be treated as a universal ALLaM/T4
context ceiling.

Both stages use TP1 utilization 0.98, FP16, eager execution, spawn and
`max_num_seqs=8`, with the same checkpoint and native wheel. P3's offline logs
record `max_num_batched_tokens=8192`; P4's online server records 2048. P3's
failed startups report 0.48 GiB KV memory and an estimated 976-token maximum;
its successful 960 run reports 0.53 GiB / 1072 cache tokens. P4 reports
1.0 GiB / 2048 cache tokens. P3 mixes reused earlier outcomes with later runs;
P4 starts its server matrix anew. GPU UUIDs also establish that P3 and P4
used different hosted allocations. These are evidenced differences. The retained
records do not isolate batching defaults, allocation history, process residency,
cleanup or execution path as the cause; no single-factor causal claim is made.

## Paper relationship and attribution

M Saiful Bari et al. (2024), *ALLaM: Large Language Models for Arabic and
English*, [arXiv:2407.15390v1](https://arxiv.org/abs/2407.15390v1), studies
Arabic/English models, second-language acquisition, vocabulary expansion,
continued/from-scratch training, bilingual data mixtures and alignment, with
Arabic and English quality evaluations. Section 2.2 explains why fragmented
Arabic tokenization increases token demand, reduces inference efficiency and
shrinks effective context under a token-count limit. This motivates bilingual
systems characterization; the small diagnostics here do not reproduce its
fertility experiment.

Section 2.4 describes 128–1024 A100 GPUs, InfiniBand, a reported 1200–1400 Gbps
node-to-node all-reduce test (also described there as RoCE), Megatron-LM,
data/tensor/pipeline parallelism, FlashAttention, bf16 and an estimated five
million GPU-hours. That training infrastructure is not comparable to constrained
dual-T4 inference. We reproduce none of its training/scaling, MMLU, MT-Bench,
human evaluation or alignment results. The paper does not validate Kaggle,
kaggle-vllm, this vLLM version or our serving results. These are independent
systems observations; nonempty bilingual responses are not language-quality
evaluation. No official endorsement is implied.

## Artifact/license boundary and limitations

The [exact-revision model card](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview/blob/a28dd1e67420cde72d3629c8633a974cf7d9c366/README.md)
declares `apache-2.0`, attributes development to NCAI/SDAIA, and refers to a
LICENSE file. The [pinned tree](https://huggingface.co/humain-ai/ALLaM-7B-Instruct-preview/tree/a28dd1e67420cde72d3629c8633a974cf7d9c366)
has no standalone LICENSE file. Accordingly this integration contains only
research-generated notebooks/scripts, synthetic prompts, generated responses,
numeric evidence, logs, telemetry, plots and identities/links. It includes no
weights, upstream tokenizer/config files, gated assets, native wheel, caches,
credentials or paper PDF. Small config facts printed by the experiments remain
historical logs, not a bundled checkpoint. This repository-content review is a
technical classification, not legal advice or permission to redistribute models.

No inference extends to arbitrary T4 hosts, A100/H100/L4, NVLink, multi-node or
production serving, other vLLM versions, or NCCL/PCIe causation. Future stronger
paper claims would require separately designed repeated sessions, randomized
order, longer workloads and independent client validation. No GPU rerun was
needed or performed for this preservation/audit.
