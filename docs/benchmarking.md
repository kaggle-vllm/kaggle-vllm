# Benchmarking methodology

No TP1/TP2 performance numbers are committed because the new harness has not
been run on the validated hardware. Functional acceptance results remain
separate from performance evidence.

Use `scripts/benchmark_kaggle.py` through the pending
[`kaggle_vllm_0_2_0_benchmark.ipynb`](../kaggle-notebooks/kaggle_vllm_0_2_0_benchmark.ipynb).
The harness records configuration, prompt/output token counts, wall and engine
latency, TTFT when upstream request metrics expose it, prefill/generation/E2E
rates, sampled per-GPU memory/utilization, `nvidia-smi`, topology and the
secret-free runtime fingerprint.

## Controlled matrix

Start with the same model, prompt, token budget, warmup and repeat count for:

1. single T4 / TP=1, eager, custom all-reduce disabled;
2. dual T4 / TP=2, same conservative settings;
3. TP=1 and TP=2 with eager disabled;
4. TP=2 with custom all-reduce enabled only if initialization is safe.

Run configurations one process at a time and confirm prior workers exit. Keep
model/cache state comparable, record topology and report each failure rather
than deleting an unfavorable configuration. Do not compare different models or
prompt lengths as if only TP changed.

Results become evidence only after the JSON files and executed notebook are
reviewed, checksummed and linked from a dated acceptance record. In particular,
do not claim TP=2 is faster, eager is optimal or custom all-reduce is beneficial
until the recorded numbers support that statement.
