# Limitations and unsupported claims

- M3 is a single fresh Kaggle dual-T4 session with five fresh-process
  repetitions; M4 has not yet been executed and remains PREPARED_FOR_KAGGLE.
- M1/M2 are observations from one Kaggle dual-T4 environment. They do not imply
  universal T4, PCIe, PHB or tensor-parallel scaling laws.
- The M3 measured-all-reduce intercept is configuration-specific and combines
  launch, synchronization, NCCL software, and transport effects. It is not
  classical alpha, universal PCIe/PHB latency, or sole-cause evidence.
- M1/M2 did not observe instantaneous scheduler/decode batch size. Concurrency
  is not used as a substitute.
- Additional-model vLLM registry support was source-inspected. SM75 execution,
  memory fit, artifact sizes and generation remain unmeasured.
- Llama and Gemma derived-artifact uploads are blocked pending redistribution
  review even when token access permits downloads.
- No Microsoft Vidur source tree/archive was found in the supplied working
  directory. M5 is therefore marked `SIMULATOR_COMPATIBILITY_LIMITATION`; no
  simulator profile or comparison was fabricated.
- Supplied nsight-python requires newer Nsight Compute/CUDA bindings than the
  validated runtime (local `ncu` is 2025.1.1 and canonical CUDA is 12.8).
  Installing it in the canonical runtime is deliberately excluded. An isolated
  auxiliary profiling environment may be evaluated later.
- GuideLLM is feasible as optional research tooling but is dependency-heavy and
  is not part of the lightweight SDK or assumed metric-equivalent to vLLM.
