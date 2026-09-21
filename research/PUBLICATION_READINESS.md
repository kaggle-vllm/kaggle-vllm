# Publication-readiness gate

Overall status: **PAPER_READY_FOR_HUMAN_REVIEW**

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
| Repository-content license review | PASS | PR #26 contains no Llama/Gemma weights, tokenizer assets, checkpoint/model-state archives, or gated upstream source; future model-artifact redistribution remains separately blocked |
| Claim discipline | PASS | Unsupported causality and universal-scaling claims are explicitly excluded |
| Paper figures | PASS | Architecture, M1–M3, and M4 crossover/latency figures are generated; M5 plots are correctly absent |
| Paper tables | PASS | Compatibility, M1–M3, M4 crossover/resource, linkage, and claim-boundary tables are generated |
| M4 Results and Discussion | PASS | Final crossover, resource-boundary, uncertainty, limitations, and conclusion text matches reviewed analyzer output |
| Manuscript editorial completion | PASS | Related work cites verified primary sources and the M1–M3 narrative is populated only from accepted evidence |

The M1--M4 technical, scientific, repository-content, and manuscript gates are
closed for human paper review. This status does not claim venue acceptance,
external artifact badging, independent replication, or legal-counsel approval.
M5 is not a PR #26 merge prerequisite; without it, the paper retains the
narrower client-specific claims in `M5_DECISION.md`. Vidur may be
`NOT_APPLICABLE_WITH_JUSTIFICATION` only after exact source inspection
establishes a scientific incompatibility; its current absence is not that
determination.

## Repository-content and model-artifact boundary

The PR diff contains 258 files. A tracked-file and type/size audit found no
`.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.gguf`, `.onnx`, model archive,
tokenizer asset, or file at least 5 MB. The applicable content boundary is:

| Class | PR #26 content | Disposition |
|---|---|---|
| Upstream model weights/assets | None | Not redistributed |
| Derived model-state/checkpoint archives | None | Not redistributed |
| Research-generated evidence | JSON/JSONL metrics, synthetic prompts, resource ledgers, logs, telemetry | Retained |
| Hashes and provenance metadata | Model revisions, file identities, manifests, reviews | Retained |
| Benchmark request/response material | Synthetic prompts and token/timing records; no generated response bodies | Retained |
| Derived tables and plots | CSV, SVG, PNG, PDF | Retained |
| Notebooks and source code | Reproducibility/orchestration/analysis sources | Retained under repository terms |

The official Meta Llama 3.2 license has a version release date of 2024-09-25
and imposes terms and notice obligations when Llama Materials or derivatives
are distributed. The official Gemma Terms of Use were last modified 2026-04-01
and impose terms, use restrictions, and notice obligations for distribution of
Gemma or Model Derivatives. Sources reviewed 2026-09-21:

- [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)
- [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

Because this PR does not distribute those model materials, its
repository-content review passes. This is a technical content classification,
not legal advice. `research/model_matrix.json` continues to block future Llama
or Gemma weight/model-state uploads until a separate review confirms the exact
artifact, terms, notice, access, and attribution obligations.
