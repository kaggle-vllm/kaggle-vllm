# Paper outline

Working title: *When Does Tensor Parallelism Pay Off on Dual T4 GPUs? Measured
Communication and Multi-Model Serving Crossovers on Kaggle*

Sections: experimental platform and immutable baseline; M1 low-load penalty;
M2 Qwen concurrency crossover; M3 collective characterization and fit limits;
M4 architecture/workload generalization; optional M5 simulator fidelity;
threats to validity; reproducibility artifacts.

Contribution boundaries: a reproducible CUDA-runtime-aware vLLM 0.18.1 platform,
controlled TP1/TP2 evidence, direct collective measurements, multi-model
crossover evidence, and open provenance. Do not claim a new engine, scheduler,
PagedAttention, CUDA kernel, or universal scaling law.
