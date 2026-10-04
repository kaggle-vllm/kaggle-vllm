# Research paper

## Characterizing Tensor-Parallel Crossover in vLLM Serving on Dual NVIDIA Tesla T4 GPUs

**Author:** Mohammad Waqas  
**Publication:** Zenodo preprint  
**Published:** 2026  
**DOI:** [10.5281/zenodo.23119478](https://doi.org/10.5281/zenodo.23119478)

📄 **[Read the research paper PDF](./kaggle-vllm-research-paper.pdf)**  
🔗 **[Zenodo publication](https://zenodo.org/records/23119478)**  
📝 **[Engineering companion article](https://waqasm86.github.io/posts/characterizing-tensor-parallel-crossover-vllm-dual-t4/)**

---

## Research question

The paper asks:

> Under which model, token-shape, and concurrency conditions does two-way
> tensor parallelism (TP2) overcome its overhead on a PHB-connected
> dual-NVIDIA-Tesla-T4 upstream-vLLM stack?

The study separates four different questions:

- **compatibility** — can the configuration execute?
- **capacity** — does the requested configuration fit the frozen resource envelope?
- **performance** — how do valid TP1 and TP2 configurations compare?
- **quality** — are generated outputs useful or correct?

The paper evaluates compatibility, capacity, and performance. It does not
claim to evaluate model-answer quality.

---

## Principal experiment

The main M4 experiment compares TP1 and TP2 across:

- 4 serving-ready model families;
- 3 exact-token workload shapes;
- 6 offered concurrency levels;
- 5 matched logical repetitions per model/workload combination;
- 720 planned fresh-server cells.

The four principal serving-ready models are:

- Qwen2.5-3B-Instruct
- Llama-3.2-3B-Instruct
- Phi-4-mini-instruct
- Ministral-3-3B-Instruct

The exact-token workloads are:

| Workload | Input tokens | Output tokens |
|---|---:|---:|
| Short / decode-heavy | 128 | 64 |
| Balanced | 512 | 256 |
| Prefill-heavy | 2048 | 128 |

The offered concurrency grid is:

```text
1, 4, 8, 16, 32, 64
```

## Main result

The first sustained favorable **TP2 output-throughput** point under the
predeclared matched-repetition rule varies by model and exact token workload:

| Principal model | Balanced | Prefill-heavy | Short |
|---|---:|---:|---:|
| Qwen2.5-3B-Instruct | 64 | N/A: resource-gated at 64 | None in tested grid |
| Phi-4-mini-instruct | 32 | 4 | 64 |
| Ministral-3-3B-Instruct | 16 | 4 | 32 |
| Llama-3.2-3B-Instruct | 16 | 4 | 32 |

These are offered-concurrency points in the tested grid, not universal TP
selection rules. Prefill-heavy work reached favorable TP2 throughput at
concurrency 4 for the three fully measurable non-Qwen models; short work
required 32 or 64. See the generated
[crossover summary](../tables/m4_crossover_summary.csv) and
[paper figures](../figures/README.md).

## Resource-gated results

All 60 planned logical shards reached a terminal reviewed state: 55 canonical
performance outcomes and five Qwen prefill-heavy resource-gated outcomes.
At concurrency 64, the frozen TP2 Qwen configuration exceeded the per-GPU
resource ceiling in all five repetitions. Its throughput and latency are
unavailable, **not zero**; no valid final TP2/TP1 performance comparison or
sustained crossover is reported for that workload. See the
[execution reconciliation](../M4_EXECUTION_RECONCILIATION.json) and
[resource summary](../tables/m4_resource_summary.csv).

## NCCL / PHB communication characterization

M3 independently measured two-rank NCCL all-reduce on the observed PHB path.
The configuration-specific fit has a 112.135079 µs intercept and 4.063916
GB/s fitted effective bandwidth; the measured effective payload bandwidth
at 64 MiB is 4.030741 GB/s. These values come from the generated
[fit](../tables/m3_communication_fit.csv) and
[statistical summary](../tables/m3_statistical_summary.csv). M3 supplies
communication context; it does not causally decompose vLLM serving or prove
that NCCL alone caused the M4 crossover.

## Supplementary ALLaM-7B case study

The [ALLaM-7B bilingual serving case study](../allam/README.md) was executed
after M4 under a different protocol. It is not part of the four-model M4
population or its crossover statistics. Nonempty Arabic and English responses
are not a model-quality evaluation.

## Reproducibility

The [research package](../README.md) documents the pinned runtime, exact-token
workloads, fresh-server cell design, five matched logical repetitions, evidence
provenance, generated [tables](../tables/README.md) and
[figures](../figures/README.md). M5 GuideLLM external validation was not run;
the claims remain specific to the tested client, topology, runtime and resource
envelope. kaggle-vllm delivers a compatible upstream-vLLM runtime and evidence
workflow; it does not implement PagedAttention, NCCL, vLLM scheduling or tensor
parallelism.

## Paper files

- [Published paper PDF](kaggle-vllm-research-paper.pdf)
- [Working manuscript](manuscript.md) and [outline](outline.md), retained as
  repository research records rather than substitutes for the published PDF

## Citation

Waqas, Mohammad (2026). *Characterizing Tensor-Parallel Crossover in vLLM
Serving on Dual NVIDIA Tesla T4 GPUs*. Zenodo.
[https://doi.org/10.5281/zenodo.23119478](https://doi.org/10.5281/zenodo.23119478).

## Canonical publication

The Zenodo DOI is the canonical public scholarly publication. The PDF in this
repository is a convenient repository-local copy associated with the software
and evidence package.
