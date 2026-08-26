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
