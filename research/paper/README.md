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
