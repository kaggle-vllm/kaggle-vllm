# Native vLLM source build

The target remains upstream vLLM v0.18.1 at `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`. A Python 3.13 native CUDA wheel has not been built or validated by this repository. The original Python 3.12 build scripts are preserved under [scripts/historical-py312](../scripts/historical-py312/README.md) for provenance, not as an active cp313 recipe.

Follow the [Python 3.13 native rebuild preparation](python313-native-runtime-rebuild.md) when working in a real Kaggle dual-T4 session. Keep source patches, exact environment and dependency findings, extension checks, inference logs, and wheel SHA256 with the new validation evidence.
