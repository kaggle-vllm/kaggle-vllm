# Tensor-parallel performance diagnostics

Milestone 1 adds a reproducible framework for measuring real upstream-vLLM
behavior on the documented Kaggle T4 x2 profile. It does not simulate model
execution or predict performance on other GPUs. Its purpose is to preserve
enough configuration, runtime, topology, telemetry, token, latency and
variability evidence to compare controlled runs without turning correlation
into an unsupported causal claim.

The framework measures the offline `vllm.LLM.generate` API. HTTP serving has
different queueing, concurrency and client/network effects and must be labeled
and measured separately.

The executed Milestone 1 session captured `nvidia-smi topo -m` reporting the
two T4s connected by `PHB`, with no `NV#` link token. NVIDIA read/write P2P
topology queries reported `OK`. These are direct command observations, not an
inference from the GPU model name or a claim about DMA behavior.

## Commands

Preview one run without importing vLLM, inspecting GPUs, downloading anything
or creating the output file:

```bash
kaggle-vllm benchmark \
  --model facebook/opt-125m \
  --model-revision 27dcfa74d334bc871f3234de431e71c6eeba5dd6 \
  --tensor-parallel-size 2 \
  --no-enforce-eager \
  --output /kaggle/working/tp2.json \
  --dry-run
```

On the validated Kaggle GPU runtime, remove `--dry-run` after bootstrap and
activation. The command refuses to overwrite an existing output or traverse a
symlinked output path. It records actual prompt/output token counts rather
than assuming every request emitted its requested maximum.

Compare two completed schema-v1 results:

```bash
kaggle-vllm compare-benchmarks \
  /kaggle/working/tp1.json \
  /kaggle/working/tp2.json \
  --baseline-label tp1 \
  --candidate-label tp2 \
  --output /kaggle/working/comparison.json
```

The comparison reports output-throughput, mean batch-latency, sampled
per-device peak-memory and optional load-time deltas. A zero or unavailable
baseline produces `null`, not an invented percentage. It also reports whether
the workload and all non-TP engine fields match.

## Reusable specification

`BenchmarkSpec` describes the model identity/representation and these engine
controls:

- tensor-parallel size;
- dtype and maximum model length;
- GPU memory utilization;
- eager versus graph execution;
- custom all-reduce enabled/disabled;
- optional `max_num_batched_tokens` and `max_num_seqs`;
- model load format and immutable revision when applicable.

`WorkloadSpec` describes fixed prompts, requested output tokens, excluded
warmup batches, measured batches, temperature, EOS behavior and seed.
Validation is CPU-only. Actual execution additionally checks the requested TP
degree against the visible GPUs.

The validated vLLM v0.18.1 source commit exposes both scheduler settings through
`EngineArgs`. The milestone passes them only when explicitly configured; it
does not silently replace upstream defaults.

## Evidence schema

Each executed run is one JSON object with `schema_version: 1` and these stable
sections:

```json
{
  "schema_version": 1,
  "status": "executed",
  "identity": {},
  "hardware": {},
  "topology": {},
  "engine": {},
  "workload": {},
  "measurements": {},
  "gpu_telemetry": {},
  "limitations": []
}
```

Identity includes UTC time, SDK version, source commit when the imported
package is inside a Git worktree, Python, OS, glibc and the measurement mode.
Hardware reuses the SDK's environment fingerprint: PyTorch/path, Torch CUDA
ABI, toolkit output, driver, GPU names/count/capabilities/memory and NCCL.

Topology preserves `nvidia-smi topo -m`, attempts the read/write peer-access
queries, and parses only observed GPU-to-GPU path tokens. A recorded `PHB`
means that the command reported a path through a PCIe host bridge. The absence
of an `NV#` token is recorded as “NVLink not observed”; it is not used to guess
DMA behavior.

Compact sampled telemetry includes per-device memory, utilization,
temperature, power and clocks when `nvidia-smi` exposes them. Missing commands
or unsupported fields remain explicit `unavailable`, `error` or `null`
observations. Raw samples are intentionally not embedded, so evidence remains
small; short peaks can therefore be missed.

## Measurements and statistics

Engine construction is timed separately. Telemetry sampling starts after
construction so no monitor thread exists while vLLM creates workers; load-time
GPU utilization is therefore not measured. Warmup batches are excluded. Every
measured batch records:

- completed request count;
- actual input and output token counts;
- total batch wall time;
- aggregate input/output tokens per second;
- request completion wall times.

Offline `generate` returns the batch after completion, so this harness uses the
containing batch duration for each request's completion-wall observation. It
does not claim per-token streaming latency or HTTP time to first token.

Aggregates include individual trial values and count, mean, median, sample
standard deviation, minimum and maximum. The overall token rates use total
tokens divided by total measured wall time. A small trial count characterizes
that session; it does not provide universal hardware precision.

## Controlled matrix

[`scripts/kaggle_tp_diagnostics.py`](../scripts/kaggle_tp_diagnostics.py)
runs each engine in an isolated child process so workers and allocations are
released before the next row. Its default OPT-125M control is deliberately
small:

1. TP=1, graph/non-eager, custom all-reduce disabled;
2. TP=2, otherwise identical;
3. TP=1, eager, custom all-reduce disabled;
4. TP=2, otherwise identical.

The historical custom-all-reduce-enabled eager TP=2 row is available through
`--include-custom-all-reduce`; it is not multiplied across every setting.
OPT-125M is a sanity/control model, not a proxy for Qwen2.5-3B performance.

When `--qwen-model` names the existing TP=2 `sharded_state`, the runner first
uses the SDK's structural/topology inspection and verifies the known four shard
names, sizes and 6,172,262,512-byte total without reading tensor bodies. It then
adds:

1. the validated eager TP=2 baseline;
2. the same TP=2 configuration with `max_num_batched_tokens=4096`.

This is an offline scheduler/batching sensitivity experiment. Qwen uses the
same four rank/part files in both runs. The TP=2 checkpoint is never used as a
TP=1 control, downloaded, rewritten or regenerated by the runner.

## Kaggle execution

The executed evidence notebook,
[`kaggle_vllm_milestone_1_tp_diagnostics.ipynb`](../kaggle-notebooks/kaggle_vllm_milestone_1_tp_diagnostics.ipynb),
uses reviewed source identity
`7e7355d64266e864a8113c30d52c612d98100350`. The self-contained notebook can
resolve that exact source archive from GitHub and the existing immutable Qwen
artifact from a Kaggle Input or its pinned Hugging Face revision. It:

1. stages only the lightweight SDK from that source under `/kaggle/working`;
2. prints a side-effect-free matrix plan;
3. runs strict profile validation, immutable bootstrap and doctor;
4. prints `nvidia-smi` and `nvidia-smi topo -m`;
5. executes each configuration in a fresh child process;
6. writes JSON, logs, topology, comparisons and `SHA256SUMS.txt` under a new
   `/kaggle/working` directory;
7. uploads nothing automatically.

The approximate bundle is:

```text
kaggle-vllm-tp-milestone-1/
  run-metadata.json
  environment.json
  topology.txt
  opt125m-*.json
  opt125m-*.log
  qwen-tp2-baseline.json
  qwen-tp2-batched-4096.json
  comparison-*.json
  summary.json
  SHA256SUMS.txt
```

The source tree, SDK staging target and evidence directory are distinct. The
notebook may replace only its explicitly named notebook-owned working tree; the
benchmark evidence writer refuses unsafe, symlinked or existing output
targets. It writes only to `/kaggle/working`, never `/kaggle/input`.

## Executed Milestone 1 evidence

The real Kaggle T4 x2 matrix completed all six planned rows on 2026-08-31 UTC
(2026-09-01 local). The
[checksummed evidence bundle](../artifacts/kaggle-2026-09-01-milestone-1/)
records Python 3.12.13, PyTorch 2.10.0+cu128, CUDA toolkit 12.8.93, driver
580.159.04, NCCL 2.27.5, two Tesla T4 GPUs at SM75, and the unchanged immutable
native runtime.

| Configuration | Aggregate output tok/s |
| --- | ---: |
| OPT-125M TP=1 graph/non-eager | 1921.17 |
| OPT-125M TP=2 graph/non-eager | 1407.60 |
| OPT-125M TP=1 eager | 312.08 |
| OPT-125M TP=2 eager | 172.32 |
| Qwen2.5-3B TP=2 baseline | 56.60 |
| Qwen2.5-3B TP=2 `max_num_batched_tokens=4096` | 54.95 |

The controlled comparisons report 26.73% lower OPT-125M TP=2 output
throughput in graph mode and 44.78% lower output throughput in eager mode.
Both TP=2 workers initialized NCCL ranks 0 and 1, sampled telemetry observed
activity on both GPUs, and runtime logs selected `TRITON_ATTN` on SM75.

The Qwen batching comparison reports a -2.91% aggregate throughput delta. Its
three baseline trials ranged from 47.80 to 63.63 output tok/s with an 8.51
tok/s sample standard deviation, so `max_num_batched_tokens=4096` did not
demonstrate a meaningful throughput improvement and the small negative delta
is not treated as a robust regression. Model-load-time differences are not
attributed to the scheduler setting because execution order and cache warming
confound that metric.

Direct observations in this evidence are throughput, latency, variability,
sampled GPU activity/memory, topology command output, runtime identity, NCCL
rank initialization, and attention-backend selection. The findings are
consistent with synchronization/communication overhead being relevant and
with the small model not recovering enough useful partitioned compute. They do
not isolate PCIe or NCCL as the sole cause. The much larger graph-versus-eager
difference also shows that execution mode materially affected this workload.

## Interpretation boundary

Direct observations include measured rates/latencies, GPU telemetry, active
GPU indices and topology command output. Examples of defensible statements:

- “TP=2 output throughput was 26.73% lower than TP=1 for the graph-mode
  OPT-125M workload.”
- “Both GPU indices had non-zero sampled utilization.”
- “`nvidia-smi topo -m` reported `PHB` and no `NV#` token.”
- “The 4096 scheduler limit did not demonstrate a meaningful throughput
  improvement in the three-repeat Qwen workload.”
- “The comparison recorded the sampled peak-memory change per device.”

Those facts can be consistent with synchronization/communication overhead, a
too-small model, scheduler/batching behavior, eager/graph effects, or multiple
interactions. They do not prove that PCIe or NCCL alone caused the delta. An
optional isolated collective microbenchmark may be added later; its bandwidth
would remain distinct from end-to-end model throughput.

TP=2 can be the correct capacity choice even when it does not improve
throughput. No result from this profile is extrapolated to A100, H100, L4,
arbitrary Kaggle GPUs, multi-node use, training or production readiness.

## Historical evidence

The original 2026-08-30 development-candidate harness and results remain
unchanged in the [0.2 benchmark report](kaggle-v0.2.0-benchmark.md), executed
[notebook](../kaggle-notebooks/kaggle_vllm_0_2_0_benchmark.ipynb), and
machine-readable artifact directory. That three-repeat OPT-125M evidence found
TP=1 faster than TP=2 and graph/non-eager faster than eager. The result is
motivation and a historical regression reference—not a newly executed
Milestone 1 result and not proof of one isolated cause.
