# Limitations and unsupported claims

- M3 is a single fresh Kaggle dual-T4 session with five fresh-process
  repetitions. M4 is in progress: the Qwen, Phi, and text-only Ministral
  short-workload concurrency-1 compatibility shards are accepted; Llama and
  Gemma canonical gates and the principal matrix remain unexecuted.
- M1/M2 are observations from one Kaggle dual-T4 environment. They do not imply
  universal T4, PCIe, PHB or tensor-parallel scaling laws.
- The M3 measured-all-reduce intercept is configuration-specific and combines
  launch, synchronization, NCCL software, and transport effects. It is not
  classical alpha, universal PCIe/PHB latency, or sole-cause evidence.
- M1/M2 did not observe instantaneous scheduler/decode batch size. Concurrency
  is not used as a substitute.
- Additional-model vLLM registry support was source-inspected. Qwen, Phi, and
  Ministral have accepted SM75 compatibility measurements; Llama and Gemma M4
  SM75 execution, memory fit, artifact sizes and generation remain unmeasured.
- Ministral compatibility covers text-only completions. Its multimodal-capable
  native implementation still performs image-encoder cache profiling at
  startup, so this evidence does not establish image-input compatibility. The
  pinned Mistral regex correction did not alter this exact prompt corpus, but
  that observation does not generalize to arbitrary text.
- The earlier Llama attempt was blocked by gated access and is not a model
  compatibility failure. Approval is user-confirmed, but canonical serving
  behavior remains unmeasured until a fresh run with the approved token passes
  audit.
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
