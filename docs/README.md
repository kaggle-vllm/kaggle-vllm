# Documentation for kaggle-vllm 0.3.2

The new v0.3 runtime targets a clean Kaggle four-NVIDIA-L4 / SM89, CPython-3.12 session; the historical T4 profile remains available. Start with the [L4 candidate](kaggle-v0.3.0-l4x4.md), [profile boundary](v0.2-to-v0.3-profile-boundary.md), [storage](runtime-storage.md), [Gemma example](gemma4.md), [installation](installation.md), [compatibility](compatibility.md), [architecture](architecture.md), [native runtime](native-runtime.md), [doctor](doctor.md), [multi-GPU](multi-gpu.md), [troubleshooting](troubleshooting.md), [release status](release.md), and [fresh Kaggle acceptance](next-kaggle-acceptance.md).

The repository also retains research records and v0.2.0 documents. Those describe the historical two-Tesla-T4/cu128 release and do not extend v0.3.0's supported profile.

For the new, output-free corrected split-v2 acceptance, see the [L4x4 notebook](../kaggle-notebooks/kaggle_vllm_0_3_0_l4x4_final_sdk_acceptance.ipynb), [fresh input guide](fresh-kaggle-v030-acceptance.md), [P31 output reuse](p31-notebook-output-reuse.md), and [Gemma workload reproducibility](gemma4-competition-reproducibility.md). The [real-model FA2 test](optional-fa2-model-acceptance.md) is deferred to a later permitted L4 environment. The core SDK remains an offline inference/runtime wrapper for compatible local models on the exact L4x4 profile.

---

## Preserved historical v0.2 documentation

The section below is retained verbatim from the v0.2 repository state for historical users. Commands and defaults above describe the v0.3 candidate.

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
- [Systems-research roadmap and milestone gates](research-roadmap.md)

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
