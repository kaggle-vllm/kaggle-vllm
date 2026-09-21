# Limitations and unsupported claims

- M3 is a single fresh Kaggle dual-T4 session with five fresh-process
  repetitions. M4 is in progress: Qwen, Phi, Llama, and text-only Ministral
  short-workload concurrency-1 compatibility shards are accepted. Final p13
  canonically preserves Gemma's pre-readiness dtype boundary. The four-model
  principal matrix is in progress, with 52 of 60 logical shards preserved.
  The first `M4-BATCH-1` allocation stopped on the Qwen prefill-heavy r00
  per-GPU VRAM guard after one new canonical shard; that failed shard remains
  review-required. All nine rows untouched in that attempt were later
  preserved by the completed Phi, Llama, and Ministral continuations. The Qwen
  r01 continuation then preserved balanced r01 and independently reproduced
  the prefill-heavy TP2/concurrency-64 resource gate. The reviewed V8
  `r02-llama` session then preserved all three Llama workloads. The V9
  `r02-ministral` session exercised M4-BATCH-3 without a terminal resource gate
  and preserved all three Ministral workloads. The V10 `r02-qwen` attempt then
  produced a third Qwen prefill-heavy TP2/concurrency-64 resource gate, but an
  outer-validator fixture mismatch stopped collection before short or balanced.
  A second physical V11 session then preserved short and balanced without
  rerunning prefill-heavy. The V12 `r02-phi` session preserved all three Phi
  workloads without a resource gate. The V13 `r03-ministral` session then
  preserved all three workloads; its final scientific archive is complete,
  but its saved executed notebook has one disclosed post-execution runner-path
  edit. Recovery is exact-hash-bound to immutable V13, the retained execution
  outputs and timestamps, and the final archive; no clean pre-edit executed
  notebook survives. The V14 `r03-qwen` allocation preserved short and
  balanced and produced a fourth prefill-heavy TP2/concurrency-64 resource
  boundary at 14,895 MiB/GPU without CUDA OOM. The V15 `r03-phi` allocation
  then preserved all three Phi workloads with no resource gate. The V16
  `r03-llama` allocation preserved all three Llama workloads with no resource
  gate. The V17 `r04-qwen` allocation then preserved balanced and short and
  produced the fifth prefill-heavy TP2/concurrency-64 resource boundary at
  14,895 MiB/GPU without CUDA OOM. The V18 `r04-phi` allocation then preserved
  all three Phi workloads with no resource gate. The corrected V20
  `r04-llama` allocation then preserved all three Llama workloads after its
  commit-consistency preflight passed. Three matrix shards are currently not
  executed. The Qwen
  sessions combine only through reviewed logical-shard reconciliation. Under the batch protocols, later shards may share one
  model/repetition physical Kaggle allocation; such shards are not fully
  independent environment realizations.
  Session/block IDs and within-session order are retained, while repetitions of
  the same model/workload are intentionally separated across allocations.
  All five planned Qwen prefill-heavy repetitions reached the same frozen
  boundary. This repetition-series completion is not a throughput value or a
  license to infer unmeasured performance, and no sixth repetition is planned.
  V9 M4-BATCH-3 continuation reduces abandonment of later same-session
  shards only after a verified terminal resource gate. It does not make those
  shards independent, remove the boundary, or predict later repetitions, and
  it applies only to post-V8 sessions and was not applied retroactively to the
  completed V8 `r02-llama` execution. The first real resource-gate exercise
  exposed the fixture mismatch before continuation guards ran. The correction
  is prospective, preserves the failed attempt, and does not change measured
  performance or benchmark parameters.
- M1/M2 are observations from one Kaggle dual-T4 environment. They do not imply
  universal T4, PCIe, PHB or tensor-parallel scaling laws.
- The M3 measured-all-reduce intercept is configuration-specific and combines
  launch, synchronization, NCCL software, and transport effects. It is not
  classical alpha, universal PCIe/PHB latency, or sole-cause evidence.
- M1/M2 did not observe instantaneous scheduler/decode batch size. Concurrency
  is not used as a substitute.
- Additional-model vLLM registry support was source-inspected. Qwen, Phi,
  Ministral, and Llama have accepted SM75 compatibility measurements. Gemma's
  canonical and diagnostic executions resolve `Gemma3ForConditionalGeneration` but stop at
  the frozen vLLM FP16 numerical-stability guard before server readiness; they
  do not measure generation, serving memory fit, latency, or throughput.
- Ministral compatibility covers text-only completions. Its multimodal-capable
  native implementation still performs image-encoder cache profiling at
  startup, so this evidence does not establish image-input compatibility. The
  pinned Mistral regex correction did not alter this exact prompt corpus, but
  that observation does not generalize to arbitrary text.
- Earlier Llama attempts blocked by access and token authorization are not
  model compatibility failures. The later clean source-equivalent run passed
  and is the only accepted Llama compatibility evidence.
- The Gemma result is scoped to `google/gemma-3-4b-it` revision
  `093f9f388b31de276ce2de164bdc2081324b9767`, the frozen FP16 M4 protocol,
  native vLLM wheel, and Tesla T4 SM75. It is not a universal Gemma 3 claim.
  FP32 was not executed: its approximately 17.20 GB weight floor exceeds both
  one T4's physical memory and the M4 hard per-GPU budget before runtime, KV
  cache, multimodal, activation, and allocator overhead, and it would define a
  materially different protocol. Gemma 4 was not substituted and remains
  out-of-scope future work requiring a new freeze and compatibility study.
- Llama and Gemma derived-artifact uploads are blocked pending redistribution
  review even when token access permits downloads.
- No Microsoft Vidur source tree/archive was found in the supplied working
  directory. M5 is therefore marked `SIMULATOR_COMPATIBILITY_LIMITATION`; no
  simulator profile or comparison was fabricated.
- Supplied nsight-python requires newer Nsight Compute/CUDA bindings than the
  validated runtime (local `ncu` is 2025.1.1 and canonical CUDA is 12.8).
  Installing it in the canonical runtime is deliberately excluded. An isolated
  auxiliary profiling environment may be evaluated later.
- GuideLLM is prepared as isolated research tooling but remains GPU-unexecuted,
  dependency-heavy, outside the lightweight SDK, and not assumed
  metric-equivalent to the primary client. The two clients' TPOT and ITL
  definitions differ explicitly.
- Multi-model TP-aware sharded-state creation is
  `OUT_OF_SCOPE_FOR_PRIMARY_M4`. M4 uses pinned Hugging Face checkpoints, so
  new Phi, Ministral, or Llama persistence archives would add GPU/storage,
  checkpoint-format, provenance, and license/redistribution obligations without
  improving the principal comparison. Llama requires gated license handling;
  Gemma never reached frozen-protocol readiness and receives no persistence
  artifact. Generating archives only after observing compatibility outcomes
  could also look post-hoc if presented as part of the original study.
- M4 exact-token generation is tokenizer-specific and synthetic. The primary
  client's event-interarrival ITL is not guaranteed to be a one-event-per-token
  engine ITL. Loopback HTTP and client scheduling remain in latency timestamps.
- The fully enumerated plan contains candidates, not promised successes. Gated
  access failures, SM75 incompatibility, OOM, or resource-guard termination must
  remain visible and may reduce the final multi-model breadth.
- The V19 r04-Llama allocation produced no benchmark observation: a stale
  notebook-embedded plan digest triggered the fail-closed source-integrity
  assertion after source checkout but before bootstrap. The queue also carried
  a stale embedded digest. Both generated files were correct at the fetched
  commit; only notebook metadata was stale. V20 repairs the generator and adds
  commit-snapshot regression checks. At that incident boundary the scientific
  state remained 49 canonical, 5 resource-gated, and 6 not executed. The
  later V20 execution is distinct: it passed the repaired invariant and
  advanced reviewed state to 52 canonical, 5 resource-gated, and 3 not
  executed.
