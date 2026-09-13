# M5 external-validation decision

Status: **INCOMPLETE; RUN ONLY AFTER M4 PRINCIPAL COMPLETION**.

M5 is optional post-M4 work under the repository roadmap and is not a merge
prerequisite for PR #26. Omitting it requires the narrower client-specific
claim boundary below.

The frozen plan contains 30 GuideLLM shards: Qwen2.5-3B, balanced workload,
concurrency 1/16/64, TP1/TP2, and five independent repetitions. GuideLLM is
pinned to commit `fc2dbe9edd4f7f1a4e9ccd752f6f43591adbcb73` in a separate client
environment. It is an independent cross-check and must validate rather than
define the primary finding.

Retain M5 as the preferred final validation milestone if Kaggle time and
storage remain after all M4 principal evidence, statistics, figures, and claims
are locked. Do not start M5 earlier and do not fabricate Vidur simulation.

The paper can stand without M5 only with narrower claims: report the finding as
replicated across fresh sessions using the frozen kaggle-vllm client, not as
client-independent validation; do not claim GuideLLM agreement, metric
portability, or external-tool robustness. GuideLLM TTFT, TPOT, and ITL
definitions must remain distinct unless raw-record reconciliation supports a
specific comparison.
