# Working manuscript: tensor-parallel crossover on dual Tesla T4

Status: **M4 results complete; publication preparation incomplete**. Remaining
bracketed text is a data-bound or editorial placeholder, not an empirical claim.

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

Motivate low-cost dual-GPU inference and the unresolved question of when TP2's
memory/compute benefits overcome collective, synchronization, scheduler, and
distributed-runtime overhead on a PCIe/PHB T4 pair.

### Contributions

- Reproducible delivery and immutable identity for a pinned upstream vLLM wheel.
- Controlled TP1/TP2 serving experiments with exact-token workloads.
- Measurement and provenance infrastructure for runtime, requests, and resources.
- Repetition-level empirical crossover characterization across four models,
  three workloads, and six concurrency points, with explicit resource outcomes.

kaggle-vllm is not a new inference engine and does not claim ownership of
PagedAttention, vLLM scheduling, tensor-parallel algorithms, or CUDA kernels.

## 2. Background

Describe vLLM serving, tensor parallelism, T4/SM75 constraints, PCIe/PHB
communication, and the distinction among throughput, latency, and capacity
crossovers.

## 3. Related work

[Add cited work on LLM serving, tensor parallelism, collective communication,
benchmark methodology, and reproducible systems research.]

## 4. Research questions

1. When does TP2 overcome TP1 overhead across model and workload regimes?
2. How do latency, throughput, capacity, and resource behavior differ?
3. Are observed changes consistent with the independently measured M3
   communication characteristics, without claiming causality from correlation?
4. Which frozen model/runtime combinations reach serving readiness on SM75?

## 5. Experimental methodology

### 5.1 Hardware and software

Populate Table 1 from `hardware_runtime.csv`: dual Tesla T4 SM75, PHB topology,
and exact Python, Torch, CUDA, NCCL, wheel, and source identities.

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

Describe source/model revisions, prompt and notebook hashes, runtime manifests,
server logs, resource guards, evidence manifests, ingestion, assembly, and
deterministic figure/table generation.

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

[Summarize only accepted M1/runtime evidence.]

### 6.2 Low-load and Qwen crossover observations

[Insert Figures 2–3 and accepted M1/M2 results.]

### 6.3 Measured NCCL/PHB communication

[Insert M3 latency/bandwidth evidence, fit diagnostics, and bounded language.]

### 6.4 Five-model compatibility gate

Insert `m4_compatibility_gate.csv`. Four models pass. Gemma 3 4B is a canonical
negative for the exact pinned revision, frozen FP16 protocol and vLLM wheel on
SM75. No requests were issued; throughput is N/A, not zero. Gemma 4 was not
substituted.

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

Address a single hosted hardware topology, exact runtime/version specificity,
synthetic tokenizer-controlled workloads, session variability, telemetry
sampling, client metric definitions, gated licenses, multimodal components,
and absence of Vidur validation. State that shards within a batch share a
physical allocation and may be correlated; retain session as a blocking factor
for limitations and justified sensitivity analysis. Do not universalize the
Gemma boundary.

## 9. Ethics and licensing

Report model licenses and gated access. Do not redistribute blocked model
weights. The benchmark uses synthetic prompts and records no user content or
credentials.

## 10. Conclusion

On the tested dual Tesla T4 platform, TP2 overcame its overhead only after a
model- and workload-dependent concurrency threshold. Prefill-heavy workloads
crossed earliest for the three fully measurable non-Qwen models; short
workloads crossed later or not at all; and Qwen prefill-heavy exposed a stable
per-GPU resource boundary instead of a valid high-concurrency performance
point. The complete evidence supports a conditional deployment rule, not a
universal preference for TP2. Publication remains subject to editorial and
license/redistribution review.

## Artifact availability

The repository retains source notebooks, protocol files, small evidence,
checksums, analysis code, and generated outputs. Large native/model artifacts
remain in documented external stores subject to identity and license controls.
The historical Qwen archive is identified by checksum but is not stored in the
repository and is not required to reproduce M4.
