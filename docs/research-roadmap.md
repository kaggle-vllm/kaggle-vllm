# Systems-research roadmap for the 0.2.0 baseline

The canonical research protocol lives in [`research/`](../research/README.md).
It extends the published 0.2.0 evidence without changing the package version or
native runtime.

- M1 is complete and records low-load TP behavior.
- M2 is complete and records the Qwen concurrency-dependent throughput crossover.
- New M3 is PREPARED_FOR_KAGGLE. Its notebook directly measures two-rank NCCL
  all-reduce rather than deriving communication coefficients from serving deltas.
- New M4 is gated on reviewed M3 evidence and successful per-model compatibility
  runs. GitHub PR #24 is governance-only and is not research M4.
- M5 is optional; the supplied workspace currently lacks the Vidur source needed
  for a defensible integration.

Use the [Kaggle execution checklist](../research/KAGGLE_EXECUTION_CHECKLIST.md).
The [experiment protocol](../research/experiment_protocol.md) defines statistics,
evidence labels and claim boundaries. The
[model matrix](../research/model_matrix.json) pins exact revisions and licensing
gates. Llama and Gemma artifact uploads remain blocked pending redistribution
review. Resource guards cap each GPU at 14.5 GiB, system RAM below 28 GiB, and
`/kaggle/working` below the practical decimal 20 GB ceiling.
The [model-artifact workflow](../research/model_artifact_workflow.md) documents
the exact save, fresh reload, API, licensing, upload-verification and cleanup
sequence.

Historical PR #23 remains historical context only. The independent M3 code does
not merge, cherry-pick, or copy its implementation. A suitable future closure
message is: “Superseded by an independent measured-collective M3 design; this PR
derived communication proxies from application throughput and is retained for
historical review only.” Do not post or close it without explicit authorization.
