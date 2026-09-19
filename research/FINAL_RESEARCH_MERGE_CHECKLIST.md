# Final research merge checklist

- [ ] All 60 active principal shards are executed in fresh T4 x2 sessions.
- [ ] Every notebook, runtime manifest, ZIP, and payload hash passes ingestion.
- [ ] All accepted evidence uses source commit `42bf096c032e2c6be1e2fa3d573c7c86ac589ba2`.
- [ ] The canonical ledger covers every planned active cell and every Gemma skip.
- [ ] Any transition refinement is justified and recorded before its execution.
- [ ] M4 assembly and paired-repetition analysis complete without missing cells.
- [ ] Figures and tables regenerate deterministically from accepted evidence.
- [ ] Gemma is N/A in performance outputs and remains in compatibility outputs.
- [ ] GuideLLM cross-check is complete, or its omission and claim reduction are explicit.
- [ ] Manuscript claims and limitations match the evidence ledger.
- [ ] License and artifact-availability statements are reviewed.
- [ ] Pytest, Ruff, static verification, Markdown links, JSON validation, and `git diff --check` pass.
- [ ] PR #26 is moved from draft only after CI is green and all scientific gates pass.
- [ ] `main` is fetched and merged normally; no research history is rewritten.
- [ ] PR #26 is merged with the repository's normal policy and exact merge SHA recorded.
- [ ] Immutable paper evidence and analysis-script identities are archived.
- [ ] A tag/release is created only if source/package changes justify one; research completion alone does not require 0.3.0.
