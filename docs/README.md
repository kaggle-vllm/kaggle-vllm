# Documentation

The root [README](../README.md) is the concise entry point. These documents keep
operational detail close to the feature or failure it describes.

## Getting started

- [Installation, bootstrap and activation](installation.md)
- [Frequently asked questions](faq.md)
- [Kaggle runtime](kaggle-runtime.md)

## Runtime and compatibility

- [Architecture and design rationale](architecture.md)
- [Compatibility contract and support matrix](compatibility.md)
- [Dependency-aware doctor](doctor.md)
- [Native runtime inspection and staging](native-runtime.md)

## Inference and serving

- [OpenAI-compatible serving](openai-serving.md)
- [Multi-GPU, tensor parallelism and NCCL](multi-gpu.md)
- [Persistent TP-aware sharded state](sharded-state.md)

## Build and reproducibility

- [Building upstream vLLM for Kaggle](building-vllm.md)
- [Source and artifact provenance](provenance.md)
- [Claim-to-evidence traceability](evidence-traceability.md)
- [Historical build findings](BUILD_FINDINGS.md)

## Benchmarking and acceptance

- [Benchmark methodology](benchmarking.md)
- [Online concurrency benchmarking](concurrency-benchmarking.md)
- [Testing: CPU CI versus Kaggle GPU acceptance](testing.md)
- [Historical validation record](validation.md)
- [Final 0.2.0 Kaggle acceptance](kaggle-v0.2.0-final-acceptance.md)
- [Final acceptance procedure and recorded result](next-kaggle-acceptance.md)

## Operations and maintenance

- [Troubleshooting](troubleshooting.md)
- [Security and supply chain](security.md)
- [Repository governance](repository-governance.md)
- [Development workflow](development.md)
- [Release process](release.md)
- [Alpha-readiness scorecard](alpha-readiness.md)

Executed historical, development-candidate, and final release-gate notebooks
are indexed under
[kaggle-notebooks](../kaggle-notebooks/README.md).
