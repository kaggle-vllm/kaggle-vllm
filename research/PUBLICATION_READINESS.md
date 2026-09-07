# Publication-readiness gate

Overall status: **NOT_PAPER_READY — M4 and external validation remain**

| Gate | Status | Evidence or blocker |
|---|---|---|
| M1 evidence | PASS | Reviewed 2026-09-01 dual-T4 evidence |
| M2 evidence | PASS | Reviewed 2026-09-02 Qwen crossover evidence |
| M3 canonical evidence | PASS | Clean-source 2026-09-07 run; canonical review and hashes pass |
| M4 breadth | FAIL | No real multi-model serving measurements yet |
| M4 independent repetitions | FAIL | No M4 GPU cells executed |
| M4 uncertainty | FAIL | Cannot estimate before repeated M4 measurements |
| External benchmark/simulator | FAIL | GuideLLM package prepared but not GPU-executed; Vidur source unavailable |
| Reproducibility | PASS | M1–M3 commands, provenance, runtime identities, and evidence retained |
| Artifact hashes | PASS | M1–M3 committed checksum manifests verify |
| License review | FAIL | Llama/Gemma redistribution remains blocked; runtime-use terms must be accepted |
| Claim discipline | PASS | Unsupported causality and universal-scaling claims are explicitly excluded |
| Paper figures | PASS | M1, M2, and M3 figures are generated; M4/M5 plots correctly absent |
| Paper tables | PASS | M1–M3 tables are generated; M4/M5 tables correctly absent |
| Manuscript skeleton | PASS | Data-bound outline exists without fabricated M4/M5 results |

The figure/table gates mean that every figure or table currently supportable by
accepted evidence is reproducible. They do not imply that the final paper's M4
or validation panels exist. Publication readiness remains blocked until M4
breadth, repetitions, uncertainty, an independent benchmark, and license review
pass. Vidur may be `NOT_APPLICABLE_WITH_JUSTIFICATION` only after exact source
inspection establishes a scientific incompatibility; its current absence is
not that determination.
