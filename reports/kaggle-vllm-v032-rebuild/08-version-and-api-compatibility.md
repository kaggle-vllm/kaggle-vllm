# Version and API compatibility

Wheel and sdist metadata checked at build time: `kaggle-vllm==0.3.2`, Python `>=3.11,<3.13`, pure `py3-none-any` SDK wheel, `kaggle-vllm = kaggle_vllm.cli_v3:main`, License and both L4/T4 package-data profiles. No `0.3.3.dev0` package metadata was found. Native vLLM remains 0.30.0 cp312-cp312 Linux x86-64 and is not bundled.

Public exports preserved: `KaggleLLM`, `KaggleSamplingParams`, `activate_runtime`, `discover_gemma4_qat`, `bootstrap`, `load_profile` and sharding helpers. Managed mode defaults were retained. Global mode is explicit and L4-only; it rejects Python 3.13, wrong distribution versions, wrong Torch CUDA, shadowed packages, missing/altered native extensions, nonlocal models and non-TP4 topology. Lazy imports preserve underlying exceptions with chaining. Historical T4 source/profile and research evidence were not removed.

Only package metadata/integrity checks were executed locally. API execution, CPU unit suites, CUDA imports and real inference remain unvalidated for this rebuilt candidate under the user's requested acceptance boundary.
