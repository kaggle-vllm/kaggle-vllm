# Limitations and unsupported claims

- M3 is a single fresh Kaggle dual-T4 session with five fresh-process
  repetitions. M4 is in progress: Qwen, Phi, Llama, and text-only Ministral
  short-workload concurrency-1 compatibility shards are accepted. Gemma has a
  reproduced diagnostic negative result but no source-equivalent executed
  notebook, so its canonicalization run and the principal matrix remain.
- M1/M2 are observations from one Kaggle dual-T4 environment. They do not imply
  universal T4, PCIe, PHB or tensor-parallel scaling laws.
- The M3 measured-all-reduce intercept is configuration-specific and combines
  launch, synchronization, NCCL software, and transport effects. It is not
  classical alpha, universal PCIe/PHB latency, or sole-cause evidence.
- M1/M2 did not observe instantaneous scheduler/decode batch size. Concurrency
  is not used as a substitute.
- Additional-model vLLM registry support was source-inspected. Qwen, Phi,
  Ministral, and Llama have accepted SM75 compatibility measurements. Gemma's
  diagnostic executions resolve `Gemma3ForConditionalGeneration` but stop at
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
- M4 exact-token generation is tokenizer-specific and synthetic. The primary
  client's event-interarrival ITL is not guaranteed to be a one-event-per-token
  engine ITL. Loopback HTTP and client scheduling remain in latency timestamps.
- The fully enumerated plan contains candidates, not promised successes. Gated
  access failures, SM75 incompatibility, OOM, or resource-guard termination must
  remain visible and may reduce the final multi-model breadth.
