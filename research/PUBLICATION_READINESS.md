# Publication-readiness gate

Overall status: **NOT_PAPER_READY — M4 and external validation remain**

| Gate | Status | Evidence or blocker |
|---|---|---|
| M1 evidence | PASS | Reviewed 2026-09-01 dual-T4 evidence |
| M2 evidence | PASS | Reviewed 2026-09-02 Qwen crossover evidence |
| M3 canonical evidence | PASS | Clean-source 2026-09-07 run; canonical review and hashes pass |
| M4 breadth | FAIL | Qwen, Phi, Llama, and Ministral compatibility are accepted; Gemma and all principal shards remain |
| M4 independent repetitions | FAIL | Compatibility has one accepted repetition per passing model; principal repetitions are unexecuted |
| M4 uncertainty | FAIL | Cannot estimate before repeated M4 measurements |
| External benchmark/simulator | FAIL | Exact isolated GuideLLM package prepared but not GPU-executed; Vidur source unavailable |
| Reproducibility | PASS | M1–M3 commands, provenance, runtime identities, and evidence retained |
| Artifact hashes | PASS | M1–M3 and accepted Qwen/Phi/Llama/Ministral M4 compatibility manifests verify; source-package hashes remain separately frozen |
| License review | FAIL | Llama/Gemma redistribution remains blocked; Gemma runtime-use terms and token authorization must be confirmed for its gate |
| Claim discipline | PASS | Unsupported causality and universal-scaling claims are explicitly excluded |
| Paper figures | PASS | M1, M2, and M3 figures are generated; M4/M5 plots correctly absent |
| Paper tables | PASS | Hardware, model, workload, M1–M3, fit, linkage, and claim-boundary tables are generated; M4/M5 result tables correctly absent |
| Manuscript skeleton | PASS | Data-bound outline exists without fabricated M4/M5 results |

The figure/table gates mean that every figure or table currently supportable by
accepted evidence is reproducible. They do not imply that the final paper's M4
or validation panels exist. Publication readiness remains blocked until M4
breadth, repetitions, uncertainty, an independent benchmark, and license review
pass. Vidur may be `NOT_APPLICABLE_WITH_JUSTIFICATION` only after exact source
inspection establishes a scientific incompatibility; its current absence is
not that determination.
