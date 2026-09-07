# Experiment protocol

## Research questions

- RQ1: What TP=2 penalty is observed at low load on the preserved dual-T4 run?
- RQ2: At what measured workloads does TP=2 outperform TP=1?
- RQ3: What are the directly measured two-rank NCCL all-reduce characteristics,
  and which application observations are consistent with them?
- RQ4: Does the crossover generalize across validated architectures and exact
  token workload shapes?
- RQ5: Can a simulator predict the measurements after explicit T4/PHB profiling?
  RQ5 is omitted if simulator mapping is scientifically invalid.

## Evidence vocabulary

`MEASURED` is a directly recorded experiment. `OBSERVED` is a qualitative fact
present in a log or measurement. `DERIVED` is deterministic analysis of measured
inputs. `ASSUMED` is an explicit experimental assumption. `HYPOTHETICAL` is a
scenario not executed. `PREDICTED` is model output. `UNSUPPORTED` means the
required evidence does not exist. Correlation and model consistency are not
causality.

## M3 primary design

Two local processes, one rank per T4, execute PyTorch `torch.distributed`
NCCL all-reduce over FP16 tensors. Payloads are 1 KiB, 4 KiB, 16 KiB, 64 KiB,
256 KiB, 1 MiB, 4 MiB, 16 MiB, 32 MiB and 64 MiB. Defaults are 20 warmups,
100 timed collectives in each of five independent repetitions. Rank-local CUDA
events time the collective; the per-iteration critical latency is the maximum
of rank 0 and rank 1. Telemetry sampling continues through the measured marker.

The fitted model is:

`T_us(S) = measured_allreduce_intercept_us + slope_us_per_wire_byte * W(S)`

For a ring all-reduce, bytes transferred per rank are
`W(S) = 2(P-1)S/P`. With `P=2`, `W(S)=S`. The inverse slope is reported as an
effective bandwidth. The intercept is runtime/topology/method specific. It is
not classical transport alpha, universal PCIe latency, or proof of causality.
Fit confidence intervals, R² and every residual are retained.

M1/M2 concurrency is never substituted for scheduler batch. Unless server
metrics expose actual running/batched sequences, M3 writes
`instantaneous_decode_batch = unobserved` and `payload_status = unknown`.

## M4 primary design

The exact machine-readable protocol is `m4_protocol.json`. Prompts are generated
and checked using the pinned tokenizer to exact lengths. Prefix caching is off;
independent runs restart the server. Every principal cell has at least five
independent repetitions and transition cells have ten. A crossover is robust
only when the paired-repetition mean-delta 95% confidence interval excludes zero
in the favorable direction. Capacity, throughput and latency crossover labels
remain distinct. Failed runs remain evidence unless a documented technical
invalidity justifies exclusion.

Use the existing M2 harness for historical continuity. Use `vllm bench serve`
as the preferred standardized cross-check. GuideLLM at commit
`fc2dbe9edd4f7f1a4e9ccd752f6f43591adbcb73`
(`v0.7.3-47-gfc2dbe9e`) is an alternate research-only cross-check;
its metric definitions must be reconciled rather than assumed identical.
