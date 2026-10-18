# Pending validation

**PASSED locally (packaging integrity only):** setuptools `build --no-isolation` produced a 0.3.2 wheel and sdist; their metadata, pure Python tag, CLI entry point, L4/T4 package data, license and absence of native binaries were inspected. Both local native wheel copies matched the expected SHA256. The packaged inference lock SHA256 is `f162e6c212371a34ed49f11885e80222fb78359a2d9a92f02b3cd29db2ce5329`.

**NOT RUN by request:** final Kaggle acceptance, four-L4 GPU checks, CUDA imports against the candidate, NCCL, TP4 engine/model generation, backend trace and post-release verification. The fresh notebook was generated and parsed, not executed.

**NOT RUN:** existing CPU unit/test suite. The user limited this task to minimum prepublication integrity checks and reserved final acceptance. Focused tests were added or adapted, but no test PASS is claimed. CUDA compilation, native wheel rebuild, pip installation, PyPI API access and publication were not performed.

**BLOCKED:** public HF-only dependency resolution and release publication pending clearance and full Hub closure. Archive ZIP hashes in the manifest are prior recorded values and will be rechecked by the offline installer on the mounted files.
