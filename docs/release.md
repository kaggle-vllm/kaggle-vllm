# 0.3.0 release status

The accepted 0.3.0 candidate wheel passed all 19 mandatory cleanup-r1 Kaggle gates in the exact four-L4 environment. This feature branch changes the SDK source and therefore needs a fresh Kaggle run before publication. The native wheel remains accepted and unchanged. See the [acceptance audit](release/v0.3.0-acceptance-audit.md).

The SDK wheel and sdist are separate from runtime-v3. No native wheel is placed inside the SDK package. Publishing, tagging, and pushing are outside this packaging task. See [release notes](../release/github-v0.3.0.md), [reproducibility](../release/reproducibility-v0.3.0.md), and [fresh acceptance](next-kaggle-acceptance.md).

The prepared [final SDK acceptance notebook](../kaggle-notebooks/kaggle_vllm_0_3_0_l4x4_final_sdk_acceptance.ipynb) defaults to the four existing corrected split-v2 Kaggle Input Datasets and a saved P31 Notebook Output. It does not require a manual runtime-v3 upload. The [additive publication plan](../release/v0.3.0-additive-publication-plan.md) preserves all historical v0.1.x/v0.2.0 GitHub, PyPI, Hugging Face binary, and model content. [Manual third-party license review](../release/v0.3.0-redistribution-review.md) remains required before any public runtime upload.

Runtime-v3 archive SHA256: `0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73` (6,633,730,070 bytes; 215 members).

---

## Preserved historical v0.2 documentation

The section below is retained verbatim from the v0.2 repository state for historical users. Commands and defaults above describe the v0.3 candidate.

# Release process

Releases deliberately separate small-SDK assurance from native GPU acceptance.

## A. SDK CI and package build

- clean reviewed branch; tests, Ruff, static identity and link checks pass;
- version, `__version__`, changelog and release notes agree;
- build with recorded tools and `SOURCE_DATE_EPOCH`;
- `twine check` passes; wheel/sdist contents contain profiles/docs expected;
- clean-venv install pulls neither vLLM, Torch nor CUDA; import has no side effect.

## B. Supply chain and licensing

- native filename, immutable HF revision and SHA256 match the profile;
- upstream tag/commit and generated distribution version are distinguished;
- overlay lock identity and dependency baseline are reviewed;
- Qwen LICENSE/NOTICE and non-commercial restriction are retained;
- no large binary, model, credential or local absolute path entered Git.

## C. Kaggle acceptance

- record Python, Torch/path, Torch CUDA, toolkit, driver, driver-reported CUDA
  maximum, T4 count/SM, NCCL, CMake and GCC;
- strict bootstrap, dependency doctor and native imports pass;
- Torch before/after identity is unchanged;
- raw NCCL and focused OPT TP=1/TP=2 checks pass;
- Qwen inspection/reload and local OpenAI endpoints pass when release scope
  requires the multi-GB model;
- new benchmark results are either executed/reviewed or explicitly pending.

## D. Publication order

1. merge reviewed SDK source;
2. build/validate the small SDK distribution;
3. publish a new PyPI version using trusted publishing;
4. run fresh Kaggle acceptance against the published identity;
5. finalize an unused Git tag and GitHub release;
6. reference, never overwrite, immutable large artifacts.

Do not republish an existing PyPI version, move public tags or replace old
release assets. CPU CI is not equivalent to GPU acceptance. The publish
workflow remains manual and main-only; no publication is part of ordinary CI.

For SDK SBOM and reproducibility guidance, see [security](security.md) and
[provenance](provenance.md).
