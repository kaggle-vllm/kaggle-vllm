# Data dictionary

## M3 raw observations

`payload_bytes` is the logical input tensor size. `rank0_latency_us` and
`rank1_latency_us` are CUDA-event durations. `critical_latency_us` is their
maximum, not their sum. `repetition` identifies independent repetition groups;
`iteration` identifies timed collectives inside a repetition. Effective GB/s is
wire bytes divided by critical seconds, using decimal GB.

Summary percentile values use nearest-rank calculation. Standard deviation over
iterations is descriptive; the 95% confidence interval uses independent
repetition means. `measured_allreduce_intercept_us` is a fitted parameter
specific to the exact experiment. `beta_effective_gb_s` is the inverse fitted
slope using decimal GB/s. `rmse_us_over_payload_means` is the square root of the
mean squared residual across the ten payload-level means.

## Application comparisons

`tp2_minus_tp1_output_tokens_per_second` is signed TP2 minus TP1 throughput.
`tp2_over_tp1_speedup` is the ratio. The M3 comparison's reciprocal-throughput
delta is DERIVED and is not isolated scheduler or per-token service time.
`unexplained_component_us` stays null unless compatible isolated service-time
evidence exists.

## M4 serving rows

`model_id` and `model_revision` identify immutable model input; the tokenizer
revision is the same pinned revision unless provenance says otherwise.
`prompt_manifest_sha256` authenticates full prompt text, UTF-8 hashes, token-ID
hashes, and exact tokenizer counts. Server-reported input/output usage must equal
the workload targets.

TTFT is request start to first content-bearing SSE event. Primary-client TPOT
is post-first-event time divided by generated-token transitions. Primary-client
`itl_ms` is mean interarrival time between content-bearing SSE events and is
explicitly an event-level approximation because one event need not equal one
token. GuideLLM ITL instead uses `(last token - first token)/(output tokens-1)`;
its TPOT includes the first token. These columns must not be silently equated.
A request-concurrency setting is never instantaneous decode batch size. Missing
optional scheduler/preemption/KV metrics are recorded as unobserved, never zero.

Each raw M4 row is one fresh-server repetition for one model, workload, TP, and
concurrency cell. `request_failures` and `oom` remain explicit. Throughput,
TTFT, TPOT, ITL, and end-to-end values may be null only for a preserved failed
cell. Resource fields are sampled and may miss sub-sample peaks.
