# Working manuscript: tensor-parallel crossover on dual Tesla T4

Status: **M1--M4 evidence and manuscript narrative complete; ready for human
paper review**. This status is not venue acceptance or legal advice. The
optional M5 cross-check remains deferred, so all claims retain the documented
client-specific boundary.

## Abstract

We present kaggle-vllm as a reproducibility and evidence-provenance layer for a
pinned upstream vLLM runtime on Kaggle dual-T4 systems. Five models entered
compatibility gating; four reached serving readiness and Gemma 3 4B encountered
a narrowly scoped FP16/SM75 boundary. Across 60 terminal logical outcomes and
720 planned fresh-server cells, TP2 crossover depended on model, workload, and
concurrency. The three non-Qwen models crossed earliest for prefill-heavy work
at concurrency 4, later for balanced work at 16 or 32, and latest for short
work at 32 or 64. Qwen crossed only for balanced work at concurrency 64; its
prefill-heavy TP2/concurrency-64 condition reproducibly exceeded the frozen
per-GPU resource ceiling. These results characterize this pinned dual-T4 stack,
not a universal tensor-parallel scaling law.

## 1. Introduction

Commodity dual-GPU environments make larger-model inference accessible, but a
second GPU is not a free throughput multiplier. Tensor parallelism divides
model work and memory while adding collectives, synchronization, scheduling,
and distributed-runtime costs. This study asks when useful parallel work
amortizes those costs on one pinned PCIe/PHB-connected Tesla T4 pair, and how
that point changes with model, token shape, and request concurrency.

### Contributions

- Reproducible delivery and immutable identity for a pinned upstream vLLM wheel.
- Controlled TP1/TP2 serving experiments with exact-token workloads.
- Measurement and provenance infrastructure for runtime, requests, and resources.
- Repetition-level empirical crossover characterization across four models,
  three workloads, and six concurrency points, with explicit resource outcomes.

kaggle-vllm is not a new inference engine and does not claim ownership of
PagedAttention, vLLM scheduling, tensor-parallel algorithms, or CUDA kernels.

## 2. Background

vLLM combines a request scheduler with block-managed KV-cache storage and GPU
execution. Tensor parallelism partitions selected model tensors across ranks;
the ranks must exchange intermediate results through collective operations.
The two tested Tesla T4 devices are SM75 GPUs connected through a PHB path,
without an observed NVLink topology token. A throughput crossover is the point
where TP2 completes output tokens faster than TP1; a latency crossover is the
corresponding favorable point for a latency metric; and a capacity or resource
boundary records whether a frozen configuration can execute within its physical
limits. These outcomes are related but are not interchangeable.

## 3. Related work

[Orca](https://www.usenix.org/conference/osdi22/presentation/yu) introduced
iteration-level scheduling and selective batching for generative Transformer
serving. [Kwon et al.](https://doi.org/10.1145/3600006.3613165) introduced
PagedAttention and the vLLM serving system, including block-based KV-cache
management and distributed workers. This repository wraps a pinned upstream
vLLM distribution; it does not reimplement or claim either contribution.

[Megatron-LM](https://arxiv.org/abs/1909.08053) established practical
intra-layer model-parallel partitioning for Transformers. NVIDIA's
[NCCL collective documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
defines the multi-rank all-reduce semantics used by the independent M3
communication measurement. Neither source predicts the crossover of this
particular serving stack, so the present work measures rather than assumes it.

The [MLPerf Inference methodology](https://arxiv.org/abs/1911.02549) motivates
explicit scenarios, metrics, and reproducible system configurations. The
[ACM artifact-review policy](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
likewise distinguishes artifact availability, functional evaluation, and
independent result validation. Consistent with those principles, this study
retains exact source/runtime/model identities, raw ledgers, failure outcomes,
and deterministic analysis, while making no claim that this repository has
received an external artifact badge or independent replication.

## 4. Research questions

1. When does TP2 overcome TP1 overhead across model and workload regimes?
2. How do latency, throughput, capacity, and resource behavior differ?
3. Are observed changes consistent with the independently measured M3
   communication characteristics, without claiming causality from correlation?
4. Which frozen model/runtime combinations reach serving readiness on SM75?

## 5. Experimental methodology

### 5.1 Hardware and software

Table 1 (`hardware_runtime.csv`) records two Tesla T4 SM75 GPUs on a PHB path,
CPython 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, NCCL 2.27.5, kaggle-vllm
0.2.0, and upstream vLLM source commit
`a26e8dc7ff2111a005144d775ecf9cebf56c45b2`. The native wheel is identified by
SHA256 `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`.

### 5.2 Workloads and metrics

Use the frozen short, balanced, and prefill-heavy exact-token workloads;
concurrency 1/4/8/16/32/64; TP1/TP2; five logical repetitions. A logical shard
is one model/workload/repetition and contains twelve serving cells, each with a
fresh vLLM server. Historical amendment `M4-BATCH-1` allowed multiple logical
shards from one repetition in one Kaggle allocation. After its first partial
batch, `M4-BATCH-2` narrowed future allocations to one model and one repetition.
After two independent Qwen prefill-heavy repetitions reached the same frozen
VRAM boundary, future-only `M4-BATCH-3` (2026-09-14; activated by V9) improved evidence-
collection reliability: a fully verified terminal resource-gated shard may be
followed by later planned shards after strict cleanup and renewed guards.
This is not a performance optimization and changes no workload, threshold,
metric, repetition, or shard order. Historical r00/r01 outer stop statuses are
reported as executed.
The V10 r02-Qwen attempt provided a third terminal boundary but exposed an
outer-validator fixture mismatch: the real monitor-caused graceful server exit
was rejected before later shards could start. Historical evidence remains
unchanged; the correction affects prospective evidence collection only and
does not alter the workload, threshold, or measured model performance.
The V11 continuation used a second physical allocation and executed only Qwen
short and balanced r02. Both are canonical; prefill-heavy was not rerun. The
two session identities remain explicit because Kaggle allocation is a blocking
variable, while reviewed reconciliation closes the single logical r02 block.
The logical repetition, physical session, continuation batch, and within-session
workload order are retained separately. Repetitions of the same model/workload
are not intentionally placed in one allocation. Define
TTFT, TPOT, event-level ITL, E2E latency, request/output/total throughput,
failures, OOM, utilization, VRAM, RAM, power, and temperature.

### 5.3 Statistics and crossover rule

The experimental unit is one logical repetition of a specific
model/workload/TP/concurrency condition, not an individual request or the whole
Kaggle allocation. Preserve raw requests but do not treat requests within a run
as independent replicates or all shards as globally independent. Use matched TP1
and TP2 repetition deltas, descriptive mean/median/standard deviation, and 95%
confidence intervals. Apply the predeclared favorable paired-mean CI rule.

### 5.4 Reproducibility and artifact handling

Every accepted result binds the source and model/tokenizer revisions, prompts,
notebook, runtime, server logs, request/resource ledgers, and archive payloads
by manifest and SHA256. Ingestion fails closed on identity, grid, semantic, or
resource inconsistencies. Reviewed shards are assembled by content address;
figures and tables are regenerated deterministically from the accepted machine
ledgers. Credentials, local caches, native wheels, and model weights are not
stored in Git.

Earlier kaggle-vllm acceptance evidence also demonstrated persistence and
reload of a TP=2 Qwen2.5-3B vLLM sharded state on the same dual-T4 runtime. The
external archive is 4,773,220,584 bytes with SHA256
`12dcb264cb74e6fa2947b5f1fbebfa14562afa2292387f49e447b3290bc0b83b`.
This is historical platform-capability evidence, not an M4 benchmark input;
M4 loads regular pinned Hugging Face checkpoints. The sharded-state mechanism
is upstream vLLM functionality and is not claimed as a novel kaggle-vllm
checkpoint format.

## 6. Results

### 6.1 Reproducible T4 platform

M1 validated the pinned runtime on two Tesla T4 SM75 GPUs with PHB topology,
no observed NVLink token, and successful bidirectional NVIDIA P2P read/write
queries. The environment used Python 3.12.13, PyTorch 2.10.0+cu128, CUDA
toolkit 12.8.93, NCCL 2.27.5, and driver 580.159.04. All six planned M1
configurations completed. These observations establish the tested platform;
they do not establish that PHB or NCCL alone caused any serving delta.

### 6.2 Low-load and Qwen crossover observations

In M1's five-trial OPT-125M controls, mean output throughput was 1921.42 versus
1408.55 tokens/s for graph-mode TP1 versus TP2, and 312.19 versus 172.33
tokens/s for eager TP1 versus TP2. TP2 was therefore operational but slower at
low load in both execution modes. The separate Qwen TP2 batching comparison
did not establish a robust improvement.

M2 then tested one Qwen2.5-3B serving matrix at concurrency
1/4/8/16/32/64 with a fresh server per cell. TP2 output throughput was below
TP1 at concurrency 1, 4, and 8, then first exceeded it at concurrency 16:
174.27 versus 138.75 tokens/s. At concurrency 32 and 64 the corresponding
values were 267.79 versus 158.65 and 311.94 versus 177.82 tokens/s. All
measured requests succeeded and no CUDA OOM occurred. This single-run M2
threshold motivated the repeated multi-model M4 design; it is not treated as a
universal concurrency rule or as repetition-level uncertainty evidence.

### 6.3 Measured NCCL/PHB communication

M3 measured two-rank NCCL all-reduce across ten payload sizes, five fresh
process repetitions, and 100 timed collectives per payload/repetition. The
5,000 critical-path observations fit the payload-mean relation
`T_us(S) = 112.135079 us + S / 4.063916 GB/s`. The fitted intercept 95% CI was
93.111210--131.158948 us and the effective-beta 95% CI was
4.051054--4.076860 GB/s; R-squared was 0.999984836 and RMSE was 20.136460 us.
The 64 MiB effective payload bandwidth was 4.030741 GB/s. The intercept is a
configuration-specific all-reduce fit parameter—not classical transport alpha
or universal PCIe/PHB latency—and the microbenchmark does not isolate the
causal contribution of communication inside vLLM serving.

### 6.4 Five-model compatibility gate

The generated `m4_compatibility_gate.csv` records four passes. Gemma 3 4B is a
canonical negative for the exact pinned revision, frozen FP16 protocol, vLLM
wheel, and SM75 device. No requests were issued; throughput is N/A, not zero.
Gemma 4 was not substituted.

### 6.5 Four-model principal crossover

The final ledger contains 55 canonical and five resource-gated outcomes: all
60 planned logical shards are terminal, covering 720 planned fresh-server
cells. Figures 6–11 and the generated M4 tables derive from the reviewed
`M4_ANALYSIS.json`. Every performance comparison uses five matched logical
repetitions; individual requests are observations within a cell, not
independent experimental replicates.

The predeclared sustained favorable 95% CI rule found matched throughput and
E2E-latency crossover points for Llama at concurrency 16 (balanced), 4
(prefill-heavy), and 32 (short); Phi at 32, 4, and 64; and Ministral at 16, 4,
and 32. Qwen crossed for balanced work only at concurrency 64. At those first
sustained throughput points, mean TP2/TP1 output-token speedups ranged from
1.095 (95% CI 1.009–1.180) for Ministral short to 1.466 (1.362–1.569) for Phi
balanced. Qwen short had no sustained crossover in the tested range.

All five Qwen prefill-heavy repetitions reached a TP2/concurrency-64 terminal
resource boundary at 14,895 MiB per physical GPU, above the frozen 14,848 MiB
ceiling, without a CUDA OOM. That cell has five resource-boundary repetitions
and zero paired performance repetitions; throughput and latency remain N/A,
never zero. No sixth repetition or post-hoc parameter adjustment was made.

The r03 Ministral post-execution notebook recovery and the V19 pre-bootstrap
Llama integrity abort remain disclosed provenance limitations. V21's final
Ministral execution passed exact source equivalence and commit consistency and
closed the matrix without changing the scientific design.

### 6.6 Mechanistic analysis

The low-concurrency TP2 penalties and later workload-specific crossovers are
consistent with fixed collective, synchronization, and distributed-runtime
costs being amortized as useful parallel work rises. Prefill-heavy work crossed
at concurrency 4 for all three fully measurable non-Qwen models, while short
work required 32 or 64. This association is consistent with, but not proved
solely by, the independently measured two-rank M3 communication regime.

### 6.7 Independent validation

The optional GuideLLM cross-check was not executed. Claims therefore remain
specific to the frozen kaggle-vllm client and its documented metric semantics;
no client-independent agreement or simulator validation is claimed.

## 7. Discussion

TP2 was not universally faster. For this stack it became favorable sooner when
the workload exposed more prefill computation, whereas short decode-heavy work
needed substantially more concurrency and Qwen never crossed within the tested
short range. Balanced crossover also varied materially by model. The repeated
Qwen prefill-heavy boundary is a capacity/resource result rather than a
performance loss: the guarded TP2 cell cannot support a valid measured
comparison under the frozen budget. Deployment decisions should therefore use
model- and workload-specific concurrency evidence and treat throughput,
latency, and resource capacity as distinct outcomes.

## 8. Threats to validity and limitations

The study observes one hosted dual-T4 topology and one exact software stack;
it does not establish behavior for other T4 hosts, interconnects, drivers,
vLLM versions, accelerators, or multi-node deployments. Tokenizer-controlled
synthetic prompts improve comparability but are not a production traffic
distribution. Shards within one batch share a physical allocation and may be
correlated; session identity and within-session order are therefore retained,
and inference uses matched repetition/session-block means rather than treating
requests as independent replicates. Sampled telemetry can miss brief peaks,
and loopback client timing includes scheduling and HTTP effects. The optional
GuideLLM/Vidur validation was not performed, so metric portability and
simulator agreement are not claimed. Ministral evidence is text-only, and the
Gemma result is restricted to the pinned revision, FP16 protocol, vLLM build,
and SM75 boundary.

## 9. Ethics and licensing

The repository code is Apache-2.0, while each upstream model retains its own
terms. A tracked-content audit found no Llama or Gemma weights, tokenizer
assets, checkpoint archive, derived model-state archive, or gated upstream
source. The retained material is research-generated evidence: synthetic
prompts, token/timing ledgers without generated response bodies, telemetry,
logs, hashes, reviews, tables, plots, notebooks, and source code. This closes
the repository-content gate for this PR; it is not legal advice or approval to
redistribute model artifacts. Any future Llama/Gemma weight or model-state
publication remains blocked pending a separate terms-and-notice review under
the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)
and [Gemma Terms of Use](https://ai.google.dev/gemma/terms). The benchmark uses
synthetic prompts and records no user content or credentials.

## 10. Conclusion

On the tested dual Tesla T4 platform, TP2 overcame its overhead only after a
model- and workload-dependent concurrency threshold. Prefill-heavy workloads
crossed earliest for the three fully measurable non-Qwen models; short
workloads crossed later or not at all; and Qwen prefill-heavy exposed a stable
per-GPU resource boundary instead of a valid high-concurrency performance
point. The complete evidence supports a conditional deployment rule, not a
universal preference for TP2. The evidence package and manuscript are ready
for human paper review; this does not constitute publication acceptance,
external artifact evaluation, or legal approval.

## Artifact availability

The repository retains source notebooks, protocol files, research-generated
requests and measurements, logs, telemetry, checksums, analysis code, and
generated tables/figures. It contains no Llama/Gemma weights, tokenizer assets,
checkpoint archives, or derived model-state archives. Large native/model
artifacts remain in documented external stores subject to identity, access,
and license controls. The historical Qwen archive is identified by checksum
but is not stored in the repository and is not required to reproduce M4.
