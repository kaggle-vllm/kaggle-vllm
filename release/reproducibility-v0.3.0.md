# 0.3.0 reproducibility

- Canonical P31 final-v2 tar SHA256: `e82085d0c6f180d2f4f8eb8f52332a92051273715e0923082fc65fc8857ddad5`
- Historical original checkpoint SHA256: `2429f6c04ebb64e6119e219cfe99e03c665c8228ec23994f2ab4f5f179e14192`
- Native vLLM 0.30.0 cp312 wheel SHA256: `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`
- vLLM source commit: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`

P31 built with `CMAKE_BUILD_TYPE=Release`, `CMAKE_CUDA_ARCHITECTURES=89`, `FETCHCONTENT_FULLY_DISCONNECTED=ON`. It derived build parallelism from `nproc --all` (48 in that session), and the source build took about 1484.674366 seconds. The SDK does not build vLLM on Kaggle.

`python3.11 scripts/build_runtime_v3.py` reads wheel metadata, evaluates Python 3.12/Linux markers and requested extras, verifies every corrected v2 wheel against `WHEELHOUSE_MANIFEST.json`, and produces an atomically promoted tar after full tar read verification. The generated dependency graph explains the retained nvdisasm distribution and excluded unreachable build wheels. Runtime-v3 is a separately distributed artifact under `dist/`, outside the SDK wheel and source-control history.

Runtime-v3 archive SHA256: `0ec85aec99b7d270511c3b62c306668d30249edc68169af5b9bad119f5c01c73` (6,633,730,070 bytes; 215 members).
