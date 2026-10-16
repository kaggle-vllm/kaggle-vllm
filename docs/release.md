# 0.3.1 release and validation status

The Python SDK 0.3.1 wheel and source distribution are already published on [PyPI](https://pypi.org/project/kaggle-vllm/0.3.1/) and in the [GitHub v0.3.1 release](https://github.com/kaggle-vllm/kaggle-vllm/releases/tag/v0.3.1). The unchanged native vLLM 0.30.0 wheel is also on [Hugging Face](https://huggingface.co/waqasm86/kaggle-vllm-binaries). Run `python scripts/release_state.py public` to recheck all three public artifact identities before any release action. Exact SDK SHA256 values are in [the release lock](../release/sdk-exact-artifacts.json); the native hash is in the L4 profile lock.

The [0.3.0 native acceptance audit](release/v0.3.0-acceptance-audit.md) remains valid for the unchanged native wheel. It does not establish integration acceptance of the published 0.3.1 SDK. The [fresh 0.3.1 notebook](../kaggle-notebooks/gemma4-competition/02_sdk_native_runtime_offline.ipynb) uses the existing 168-wheel P33 lock, corrected split-v2 Inputs, the frozen native wheel, and the licensed Gemma 4 Input in a new offline four-L4 session. Its actual 19-gate output determines 0.3.1 integration status.

The exact published distributions must never be rebuilt or uploaded under the same version. GitHub CI already skips PyPI staging and publication when the live files match the lock; an identity mismatch fails closed. Editing documentation does not require a new GPU run. A change to SDK runtime behavior needs relevant SDK integration evidence; a change to the native wheel would need native hardware acceptance. Gemma TP4 checkpoint research and model publication have their own save, reload, completeness, licensing, and hash checks and do not gate the already published Python SDK.

The [third-party redistribution review](../release/v0.3.0-redistribution-review.md) remains open for any public mirror of the 168-wheel runtime or broader 195-wheel archival closure. Public SDK and native-wheel availability does not clear those separate third-party notice obligations. Do not use the archival 6.6 GB runtime-v3 archive as the default Kaggle Input.

---

## Preserved historical v0.2 documentation

The section below records the earlier v0.2 release process. Its publication order and T4 checks are historical; the 0.3.1 state above governs current work.

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
