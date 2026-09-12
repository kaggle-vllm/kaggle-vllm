# Publication-readiness gate

Overall status: **NOT_PAPER_READY — M4 principal evidence remains**

| Gate | Status | Evidence or blocker |
|---|---|---|
| M1 evidence | PASS | Reviewed 2026-09-01 dual-T4 evidence |
| M2 evidence | PASS | Reviewed 2026-09-02 Qwen crossover evidence |
| M3 canonical evidence | PASS | Clean-source 2026-09-07 run; canonical review and hashes pass |
| M4 compatibility breadth | PASS | Frozen five-model gate closed: four passes and one canonical, narrowly scoped Gemma negative |
| M4 principal breadth | FAIL | 9/60 four-model principal shards are preserved; one reviewed resource boundary and 50 not-executed shards remain |
| M4 logical repetitions | FAIL | Qwen-short has 2/5 preserved; no condition has the complete five repetitions |
| M4 uncertainty | FAIL | Cannot estimate before repeated M4 measurements |
| External benchmark/simulator | OPTIONAL POST-M4 | M5 is optional under the roadmap; omit client-independent claims unless GuideLLM is actually executed |
| Reproducibility | PASS | M1–M3 commands, provenance, runtime identities, and evidence retained |
| Artifact hashes | PASS | M1–M3 and all five M4 compatibility manifests verify; source-package hashes remain separately frozen |
| License review | FAIL | Llama/Gemma redistribution remains blocked even though authenticated compatibility execution succeeded |
| Claim discipline | PASS | Unsupported causality and universal-scaling claims are explicitly excluded |
| Paper figures | PASS | Architecture and M1–M3 figures are generated; M4 principal/M5 plots correctly absent |
| Paper tables | PASS | Compatibility plus hardware, model, workload, M1–M3, fit, linkage, and claim-boundary tables are generated; M4 principal/M5 result tables correctly absent |
| Manuscript skeleton | PASS | A section-complete data-bound scaffold exists without fabricated M4/M5 results |

The figure/table gates mean that every figure or table currently supportable by
accepted evidence is reproducible. They do not imply that the final paper's M4
or validation panels exist. Publication readiness remains blocked until M4
breadth, repetitions, uncertainty, and license review pass. M5 is not a PR #26
merge prerequisite; without it, the final paper must retain the narrower
client-specific claims in `M5_DECISION.md`. Vidur may be
`NOT_APPLICABLE_WITH_JUSTIFICATION` only after exact source inspection
establishes a scientific incompatibility; its current absence is not that
determination.
