# Milestone 2 evidence — GPU execution pending

This directory is reserved for reviewed evidence from the online
`Qwen/Qwen2.5-3B-Instruct` TP1/TP2 concurrency matrix on a real Kaggle dual
Tesla T4 session.

GPU execution is pending. There are no synthetic, simulated, or locally
fabricated result JSON files here. No throughput crossover, capacity
crossover, OOM, KV-cache saturation, or Kaggle acceptance result is claimed.

The output-free source notebook is
[`kaggle_vllm_milestone_2_concurrency_crossover.ipynb`](../../kaggle-notebooks/kaggle_vllm_milestone_2_concurrency_crossover.ipynb).
It will produce `/kaggle/working/kaggle-vllm-milestone-2.zip`. After real
execution, that ZIP must be downloaded, its printed SHA256 checked, and every
cell/log/schema/checksum reviewed before selected Git-safe evidence is added
here.

Expected reviewed bundle contents are documented in
[`docs/concurrency-benchmarking.md`](../../docs/concurrency-benchmarking.md).
