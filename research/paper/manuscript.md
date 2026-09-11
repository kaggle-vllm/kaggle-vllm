# Working manuscript: tensor-parallel crossover on dual Tesla T4

Status: **results incomplete**. Bracketed text is a data-bound placeholder, not
an empirical claim.

## Abstract

We present kaggle-vllm as a reproducibility and evidence-provenance layer for a
pinned upstream vLLM runtime on Kaggle dual-T4 systems. [Insert only reviewed M4
crossover findings and uncertainty.] Five models entered compatibility gating;
four reached serving readiness and one encountered a narrowly scoped
precision/hardware boundary. [Insert conclusions only after principal analysis.]

## 1. Introduction

Motivate low-cost dual-GPU inference and the unresolved question of when TP2's
memory/compute benefits overcome collective, synchronization, scheduler, and
distributed-runtime overhead on a PCIe/PHB T4 pair.

### Contributions

- Reproducible delivery and immutable identity for a pinned upstream vLLM wheel.
- Controlled TP1/TP2 serving experiments with exact-token workloads.
- Measurement and provenance infrastructure for runtime, requests, and resources.
- Empirical crossover characterization [pending M4].

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

[Blocked until 60 principal shards and any justified refinements are accepted.]
Insert per-model curves, speedup heatmap, latency tradeoff, crossover and
resource tables only from `M4_ANALYSIS.json`.

### 6.6 Mechanistic analysis

[Relate M3 communication observations to M4 behavior as consistency evidence;
do not infer sole causality.]

### 6.7 Independent validation

[Blocked on reviewed GuideLLM evidence, or explicitly scope this to future work.]

## 7. Discussion

[Interpret workload/model differences only after M4. Distinguish throughput,
latency and capacity crossovers and practical deployment implications.]

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

[Blocked until M4 Results and Discussion pass the publication-readiness gate.]

## Artifact availability

The repository retains source notebooks, protocol files, small evidence,
checksums, analysis code, and generated outputs. Large native/model artifacts
remain in documented external stores subject to identity and license controls.
The historical Qwen archive is identified by checksum but is not stored in the
repository and is not required to reproduce M4.
