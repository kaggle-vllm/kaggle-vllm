# Native runtime status

The active `kaggle-t4x2-cu128` profile has `status: target_unvalidated`, ABI `cp313`, and null native wheel filename, SHA256, Hub revision, and overlay lock. Bootstrap cannot download or install a native runtime until real Kaggle dual-T4 validation supplies those fields.

The historical CPython 3.12 wheel identity and checksum remain in [historical profile](../compat/historical-py312-v0.2.0/profile.json), `release/`, and the original acceptance evidence. The historical wheel is not compatible with Python 3.13 and is rejected by direct wheel staging.

See [native rebuild preparation](python313-native-runtime-rebuild.md) for the checks needed before enabling cp313 bootstrap.
