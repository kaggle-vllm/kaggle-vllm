# Benchmarking methodology

The 0.2 development-candidate harness was executed on the validated hardware on
2026-08-30. Functional acceptance remains separate from this performance
evidence. The exact source and results are retained in the
[`0.2 benchmark report`](kaggle-v0.2.0-benchmark.md), executed
[`kaggle_vllm_0_2_0_benchmark.ipynb`](../kaggle-notebooks/kaggle_vllm_0_2_0_benchmark.ipynb)
and machine-readable artifact directory.
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

The recorded results show TP=1 faster than TP=2 for OPT-125M, where NCCL
communication overhead dominates. They also show non-eager execution faster
than eager execution in this small controlled workload. With only three
measured repeats per configuration, small custom-all-reduce differences are not
treated as robust improvements or regressions. These numbers do not imply that
TP=2 is slower for larger capacity-driven workloads or that any setting is
universally optimal.
