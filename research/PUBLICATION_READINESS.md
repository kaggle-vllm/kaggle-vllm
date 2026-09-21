# Publication-readiness gate

Overall status: **NOT_PAPER_READY — human license and editorial review remain**

| Gate | Status | Evidence or blocker |
|---|---|---|
| M1 evidence | PASS | Reviewed 2026-09-01 dual-T4 evidence |
| M2 evidence | PASS | Reviewed 2026-09-02 Qwen crossover evidence |
| M3 canonical evidence | PASS | Clean-source 2026-09-07 run; canonical review and hashes pass |
| M4 compatibility breadth | PASS | Frozen five-model gate closed: four passes and one canonical, narrowly scoped Gemma negative |
| M4 principal breadth | PASS | 55 canonical plus 5 reviewed resource-gated outcomes settle all 60 logical shards / 720 planned cells; zero remain queued |
| M4 logical repetitions | PASS | Every model/workload has five terminal repetitions; Qwen prefill-heavy has five resource-boundary outcomes rather than fabricated performance |
| M4 uncertainty | PASS | Matched five-repetition summaries and 95% CIs are generated; requests are not treated as independent repetitions |
| External benchmark/simulator | OPTIONAL DEFERRED | M5 was not run; claims are explicitly client-specific and do not assert GuideLLM or simulator agreement |
| Reproducibility | PASS | M1–M4 commands, provenance, runtime identities, reviewed assembly, and deterministic outputs are retained |
| Artifact hashes | PASS | M1–M4 manifests verify; the final 60-shard assembled package is checksummed and source-package hashes remain separately frozen |
| License review | FAIL | Llama/Gemma redistribution remains blocked even though authenticated compatibility execution succeeded |
| Claim discipline | PASS | Unsupported causality and universal-scaling claims are explicitly excluded |
| Paper figures | PASS | Architecture, M1–M3, and M4 crossover/latency figures are generated; M5 plots are correctly absent |
| Paper tables | PASS | Compatibility, M1–M3, M4 crossover/resource, linkage, and claim-boundary tables are generated |
| M4 Results and Discussion | PASS | Final crossover, resource-boundary, uncertainty, limitations, and conclusion text matches reviewed analyzer output |
| Manuscript editorial completion | FAIL | Related-work citations and remaining M1–M3 narrative placeholders require human scholarly/editorial completion |

The M4 technical and scientific gates are closed: all planned logical outcomes
are terminal, the final repetition-level analysis exists, and its figures and
tables are reproducible. Publication readiness remains blocked by human review
of Llama/Gemma redistribution terms and by ordinary scholarly/editorial work,
including related-work citations and remaining M1–M3 narrative placeholders.
M5 is not a PR #26 merge prerequisite; without it, the paper retains the
narrower client-specific claims in `M5_DECISION.md`. Vidur may be
`NOT_APPLICABLE_WITH_JUSTIFICATION` only after exact source inspection
establishes a scientific incompatibility; its current absence is not that
determination.
