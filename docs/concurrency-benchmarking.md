# Online concurrency benchmarking

Milestone 2 asks a measurement question: at which tested request concurrency,
if any, does TP=2 become preferable to TP=1 for
`Qwen/Qwen2.5-3B-Instruct` on Kaggle's two Tesla T4 GPUs? The benchmark does
not assume a crossover, OOM, or KV-cache saturation will occur.

This is an online serving benchmark. It sends simultaneous HTTP requests to
the upstream vLLM OpenAI-compatible server. It is intentionally separate from
the Milestone 1 offline `vllm.LLM.generate` benchmark; an offline batch size is
not treated as request concurrency and cannot supply streaming TTFT.

## Controlled model and engine

The default model is the original Transformers representation of
`Qwen/Qwen2.5-3B-Instruct`, pinned to Hugging Face revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. The same model identity, revision,
dtype, scheduler controls, prompt corpus, token limits, EOS policy, seed, and
generation parameters are used for TP1 and TP2. An attached read-only Kaggle
Input may replace the Hub download only when it is the same Transformers
snapshot. The runner rejects vLLM `model-rank-*-part-*` files: the existing
TP=2 `sharded_state` is topology-specific and is never used as the TP1 model.

Default engine controls are FP16, eager execution, custom all-reduce disabled,
`max_model_len=4096`, `max_num_seqs=64`, and
`gpu_memory_utilization=0.90`. Prefix caching is explicitly disabled so repeated
fixed-corpus requests cannot silently avoid their controlled prefill work. The
fixed corpus repeats a technical context to create a reproducible long-context
load; actual input counts come from server usage rather than a guessed tokenizer
count. Each request asks for 512 output tokens with a 1,800-second timeout.
These conservative T4-compatible settings are
recorded controls, not universal performance recommendations. The workload is
representative and memory-relevant but is not manipulated to force an OOM.

## Matrix and concurrency semantics

The required matrix is:

```text
TP=1 × concurrency 1,4,8,16,32,64
TP=2 × concurrency 1,4,8,16,32,64
```

Concurrency is the maximum number of HTTP requests simultaneously in flight.
The client uses a closed-loop worker pool; queued client work has not yet
started a request. Overall throughput includes the full measured interval
from submission of the first wave through completion of the final wave.

By default, each cell has at least 20 measured requests and at least three
waves: `max(20, 3 × concurrency)`. One full-concurrency warmup wave is excluded
by default. The
total request count, warmup count, fixed prompt corpus/profile, requested
output tokens, temperature, EOS policy, seed, and timeout are explicit CLI
settings. A request count too small for three waves or 20 observations is
rejected rather than used to produce a weak p95.

The default deterministic order interleaves TP1 and TP2 at each concurrency.
`--matrix-order tp-major` provides TP1 c1→c64 followed by TP2 c1→c64. The order
is written to `run-metadata.json`. The scientific default starts a fresh
server for every `(TP, concurrency)` cell. Server/model load time is excluded
from TTFT, TPOT, and throughput. Per-cell isolation reduces allocation,
fragmentation, and earlier-load contamination at the cost of Kaggle runtime.

For TP1, the server child receives `CUDA_VISIBLE_DEVICES=0`; TP2 receives
`CUDA_VISIBLE_DEVICES=0,1`. Evidence records those physical indices. The two
cards provide approximately 30 GB aggregate device memory across two T4 GPUs;
they are not one fungible 30 GB GPU.

## Request and metric definitions

Every measured request is written to a JSONL ledger with request ID,
concurrency, UTC start/first-token/completion timestamps, server-reported
input/output token counts, TTFT, end-to-end latency, TPOT, HTTP status, neutral
status, and any bounded error detail.

- **TTFT** is `first content-bearing streamed SSE event timestamp - request
  start timestamp`. Role-only, comment, usage, and `[DONE]` events are not
  tokens and do not establish TTFT. Setup and server load precede dispatch.
- **TPOT**, for more than one actual output token, is `(completion timestamp -
  first-token timestamp) / (output_token_count - 1)`. It is `null` for zero or
  one output token, missing usage, or a failed request.
- **Output throughput** is `successful actual output tokens / measured
  wall-clock interval`. Failed or usage-incomplete requests contribute no
  invented tokens; their failure counts remain explicit.
- Percentiles use the nearest-rank method: sort `n` values and select one-based
  rank `ceil(p/100 × n)`. Each distribution reports count, mean, median,
  p50/p95/p99, sample standard deviation, minimum, and maximum. Empty or
  undefined values are `null`, never zero placeholders.

The streaming request asks the server for `stream_options.include_usage=true`.
The usage event is the source of actual prompt and completion token counts.
A stream without a content event and complete usage is recorded as a failure,
because required token/latency evidence is incomplete.

## Failures and OOM interpretation

Statuses are neutral observations: `completed`, `client_timeout`,
`HTTP_error`, `connection_error`, `server_exit`, `server_start_failure`,
`CUDA_OOM_observed`, `request_failure`, or `unknown_failure`.

CUDA OOM is assigned only when request/error/server-log text matches an actual
CUDA out-of-memory exception. Lower throughput, increased TTFT, failed
readiness, or a full sampled memory value is not enough. Likewise, the report
does not claim host KV-cache swapping unless the captured runtime metrics or
logs explicitly show it. The default experiment does not enable CPU swap to
manufacture a capacity result.

Startup failure and unexpected client exceptions still produce the cell JSON,
server log, request JSONL, and an unavailable metrics snapshot. Other cells
continue, making partial evidence reviewable.

## GPU and Prometheus telemetry

`nvidia-smi` sampling begins immediately before measured request submission
and ends after the final request. Each available GPU records used and total
memory, GPU and memory utilization, power, SM/memory clocks, and temperature.
Evidence includes monitor start/end timestamps, total sample count, per-GPU
peak sampled memory, and maximum aggregate memory across a simultaneous
sample. Sampling can miss instantaneous peaks.

The runner preserves raw `/metrics` text immediately before and after every
measured interval. It parses only metric families verified in upstream vLLM
source commit `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`: running/waiting
requests, KV-cache usage, preemptions, prompt/generation tokens, and successful
request counts. A family absent from the installed endpoint stays absent or
empty. Before/after snapshots do not claim to capture an in-flight gauge peak.

## Commands

Preview one client cell without importing vLLM, inspecting hardware,
downloading a model, opening a network connection, or writing output:

```bash
kaggle-vllm benchmark-serving \
  --model Qwen/Qwen2.5-3B-Instruct \
  --model-revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --tensor-parallel-size 1 \
  --concurrency 16 \
  --output /kaggle/working/qwen-tp1-c16.json \
  --dry-run
```

Preview the complete matrix:

```bash
python scripts/kaggle_concurrency_crossover.py \
  --output-dir /kaggle/working/kaggle-vllm-milestone-2 \
  --dry-run
```

Executed evidence requires the reviewed source identity:

```bash
python scripts/kaggle_concurrency_crossover.py \
  --source-identity REVIEWED_COMMIT \
  --output-dir /kaggle/working/kaggle-vllm-milestone-2
```

The output-free
[`kaggle_vllm_milestone_2_concurrency_crossover.ipynb`](../kaggle-notebooks/kaggle_vllm_milestone_2_concurrency_crossover.ipynb)
stages the pinned SDK source, bootstraps the unchanged immutable native wheel,
runs strict doctor, resolves the pinned model, prints the dry-run plan, runs
all cells, creates checksums, and makes a ZIP. It writes only below
`/kaggle/working`; it does not modify `/kaggle/input`, publish data, rebuild
vLLM, or regenerate model weights.

## Executed dual-T4 results

The 2026-09-03 Kaggle execution used benchmark source
`4f8dcc1c032d65d54b1cce3ca213535d68fd5099`, the pinned Qwen revision above,
and the documented native runtime on two Tesla T4 GPUs. The full checksummed
evidence is under
[`artifacts/kaggle-2026-09-02-milestone-2/`](../artifacts/kaggle-2026-09-02-milestone-2/).

| Concurrency | TP1 output tok/s | TP2 output tok/s | TP2 delta | TP1 p95 TTFT (s) | TP2 p95 TTFT (s) | TP1 p95 TPOT (s) | TP2 p95 TPOT (s) | TP1 success/fail | TP2 success/fail |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.29 | 14.95 | -43.1% | 1.606 | 0.950 | 0.0353 | 0.0663 | 20/0 | 20/0 |
| 4 | 76.87 | 55.21 | -28.2% | 5.862 | 3.223 | 0.0463 | 0.0697 | 20/0 | 20/0 |
| 8 | 108.53 | 99.80 | -8.0% | 11.696 | 6.293 | 0.0677 | 0.0792 | 24/0 | 24/0 |
| 16 | 138.75 | 174.27 | +25.6% | 21.906 | 11.876 | 0.1086 | 0.0887 | 48/0 | 48/0 |
| 32 | 158.65 | 267.79 | +68.8% | 43.826 | 23.503 | 0.1920 | 0.1138 | 96/0 | 96/0 |
| 64 | 177.82 | 311.94 | +75.4% | 84.778 | 46.288 | 0.3452 | 0.1988 | 192/0 | 192/0 |

TP1 delivered higher output throughput at concurrency 1, 4, and 8. The first
observed throughput crossover was concurrency 16; TP2's measured advantage
then increased at 32 and 64. Every measured request succeeded. No CUDA OOM
was observed for either TP degree, and no TP1-fails/TP2-survives capacity
crossover occurred through concurrency 64.

At concurrency 64, peak sampled memory was 14,255 MiB on TP1's physical GPU
0. TP2 reached 14,701 MiB sampled on each device, or 29,402 MiB
(approximately 29.4 GiB) aggregate sampled memory across the two T4s. This is
not one monolithic 30 GB GPU, and high sampled utilization is not an OOM.

These results are specific to the recorded model, workload, engine controls,
runtime, and hardware. Concurrency 16 is not a universal TP-selection
threshold, and the evidence does not assign the crossover to PHB topology or
memory behavior alone.

## Evidence and crossover analysis

Cell results use schema `kaggle-vllm-serving-benchmark-v1`. The summary uses
`kaggle-vllm-serving-summary-v1`. Predictable per-cell files include `.json`,
`.server.log`, `.metrics.txt`, `.telemetry.jsonl`, and `-requests.jsonl`; the bundle also contains
`run-metadata.json`, `environment.json`, `topology.txt`, `summary.json`, and
`SHA256SUMS.txt`.

The throughput crossover is the first tested concurrency where TP2 output
throughput exceeds TP1 while TP2 maintains an equal or higher request success
rate. The capacity crossover is independently the first tested concurrency
where TP1 has request failures and TP2 completes every request. The summary
also reports each TP mode's first failure and first evidenced CUDA OOM.

Every crossover field may be `null`. “No crossover observed in the tested
range,” “both configurations survive,” and “both fail” are valid outcomes.
Throughput and capacity crossovers are never conflated, and no causal claim is
made from PHB topology alone.

## Acceptance boundary

CPU tests validate request parsing, formulas, schema, safety, lifecycle, and
analysis. They do not validate T4 execution. Milestone 2 GPU acceptance is
**PASS** for the reviewed, checksummed 2026-09-03 Kaggle T4×2 run. That status
applies only to the preserved execution evidence and does not turn CPU dry
runs into GPU evidence or extend the result beyond the tested matrix.
