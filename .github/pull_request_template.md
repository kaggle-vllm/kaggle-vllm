## Problem and scope

Describe the user-visible problem and whether this changes only the lightweight
SDK or also requires future Kaggle GPU acceptance.

## Evidence and validation

- [ ] CPU tests pass
- [ ] Ruff/static/link/package checks pass as applicable
- [ ] No import-time network/install/activation side effect
- [ ] No vLLM/Torch/CUDA normal dependency added
- [ ] Profile/provenance/checksum changes are consistent
- [ ] GPU claims link real evidence or are explicitly pending
- [ ] No large binary/model/cache/log or credential added
- [ ] Upstream and model attribution/licensing remain accurate

Commands and exact results:

## Release impact

State version/changelog implications and confirm no published tag, PyPI version
or historical artifact is being overwritten.
