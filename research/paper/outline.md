# Paper outline

Preferred title: *Characterizing Tensor-Parallel LLM Serving on Dual NVIDIA T4
GPUs*

Alternative title: *When Two GPUs Beat One: Characterizing Tensor-Parallel LLM
Serving on Commodity Turing GPUs*

## Abstract skeleton

Motivation: tensor parallelism increases aggregate compute and memory capacity
but introduces collective communication and coordination costs. Method: use a
pinned vLLM 0.18.1 runtime on dual PHB-connected Tesla T4 GPUs, preserve
provenance, measure low-load and concurrency behavior, directly characterize
two-rank NCCL all-reduce, and test generalization across model architectures and
token shapes. Results: insert generated M1/M2/M3 facts and M4/independent-
validation results only when their machine-readable gates pass. Contribution:
a reproducible empirical characterization and open evidence package, not a new
inference engine or scaling law.

## 1. Introduction and motivation

- State the practical TP=1 versus TP=2 deployment decision.
- Explain why low-load penalty, concurrency crossover, latency, throughput, and
  capacity crossover are distinct.
- Research question: under which serving workloads does dual-T4 tensor
  parallelism overcome its communication overhead, and how does that boundary
  vary by architecture and workload shape?

## 2. Background and related work

- Tensor parallel transformer execution and collective operations.
- Continuous batching, prefill/decode asymmetry, and serving metrics.
- Empirical inference benchmarking and simulator-based prediction.
- Position this artifact as measurement and reproducibility work.

## 3. Experimental methodology

- Hardware/runtime table generated from provenance.
- Exact model/tokenizer revisions and license/access table.
- Exact SHORT, BALANCED, and PREFILL_HEAVY token-controlled workloads.
- TP=1/TP=2, concurrency matrix, repetitions, server reset, disabled prefix
  caching, resource guards, failure preservation, and statistical methods.
- GuideLLM metric-reconciliation protocol; Vidur only if compatibility is
  defensible after source inspection.

## 4. Reproducibility

- Repository commit, notebook, native wheel, evidence, and model hashes.
- Kaggle environment, topology, Torch/CUDA/NCCL, commands, seeds, telemetry,
  and immutable raw ledgers.
- Machine generation of every numerical paper figure/table.

## 5. Results

### 5.1 M1 low-load behavior

Insert generated Figure 1 and M1 table. Restrict interpretation to the observed
models, modes, environment, and five trials per cell.

### 5.2 M2 Qwen concurrency crossover

Insert generated Figure 2 and M2 table. Label the concurrency-16 throughput
crossover as an observed single-run matrix result, not a universal threshold.

### 5.3 M3 collective characterization

Insert generated Figures 3–5 and the M3 statistical table. Report the
configuration-specific measured-all-reduce intercept and effective beta with
confidence intervals, residuals, R², and RMSE. Do not rename the intercept to
classical alpha or universal PCIe/PHB latency.

### 5.4 M4 model/workload generalization

Placeholder blocked on accepted M4 evidence. Populate per-model crossover
curves, speedup heatmap, TTFT, TPOT, failure/capacity outcomes, and uncertainty
only from the M4 analyzer output.

### 5.5 Independent validation

Placeholder blocked on a real GuideLLM cross-check or defensible simulator
comparison. Report metric-definition differences and prediction/measurement
errors rather than treating tools as equivalent.

## 6. Discussion

Interpret workload-dependent tradeoffs only after M4. Separate throughput,
latency, and capacity crossovers. Discuss which M3 observations are consistent
with serving behavior without causal attribution.

## 7. Limitations and threats to validity

Single hosted hardware topology; runtime/version specificity; model selection;
tokenizer-controlled synthetic workload limits; session variability; limited M2
repetition; collective microbenchmark versus serving-path gap; unavailable or
incompatible simulator risk; gated-model access and redistribution constraints.

## 8. Conclusion

Summarize only claims that pass `PUBLICATION_READINESS.md`. Do not generalize to
all T4, PCIe, PHB, vLLM, model, or serving environments.

## Artifact availability statement

Source notebooks, CPU analysis code, small evidence ledgers, provenance,
checksums, and generated figures/tables are retained in the repository. Large
native/model artifacts remain in their documented external stores subject to
license and redistribution constraints. Secrets and gated weights are excluded.

## Claim boundary

Allowed framing: reproducible vLLM 0.18.1 dual-T4 execution and measurement;
empirical TP1/TP2 characterization; directly measured collective regime;
workload-dependent crossover; cross-model generalization after M4; open
notebooks/evidence/provenance. Excluded framing: new engine, scheduler, CUDA
kernel, PagedAttention implementation, or universal scaling law.
