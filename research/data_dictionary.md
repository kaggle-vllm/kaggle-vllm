# Data dictionary

## M3 raw observations

`payload_bytes` is the logical input tensor size. `rank0_latency_us` and
`rank1_latency_us` are CUDA-event durations. `critical_latency_us` is their
maximum, not their sum. `repetition` identifies independent repetition groups;
`iteration` identifies timed collectives inside a repetition. Effective GB/s is
wire bytes divided by critical seconds, using decimal GB.

Summary percentile values use nearest-rank calculation. Standard deviation over
iterations is descriptive; the 95% confidence interval uses independent
repetition means. `measured_allreduce_intercept_us` is a fitted proxy specific to
the exact experiment.

## Application comparisons

`tp2_minus_tp1_output_tokens_per_second` is signed TP2 minus TP1 throughput.
`tp2_over_tp1_speedup` is the ratio. The M3 comparison's reciprocal-throughput
delta is DERIVED and is not isolated scheduler or per-token service time.
`unexplained_component_us` stays null unless compatible isolated service-time
evidence exists.

## M4 serving rows

Token counts are tokenizer-observed exact counts. TTFT is time to first token;
TPOT is post-first-token time divided by generated-token transitions; ITL is
the distribution of adjacent output-token intervals. Each benchmark tool must
record its own definition. A request-concurrency setting is never instantaneous
decode batch size. Missing optional scheduler/preemption/KV metrics are recorded
as unobserved, never zero.
