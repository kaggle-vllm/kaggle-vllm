# Development workflow

The SDK is intentionally small and GPU-free for ordinary development.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests examples scripts
```

Do not install the native wheel globally. Do not add vLLM, Torch, CUDA or NCCL
as normal SDK dependencies. Importing the package must remain side-effect free.

## Changing a profile

Update the packaged profile, dependency baseline, overlay requirements/lock,
compatibility JSON and consolidated provenance together. Record the exact
upstream requirements or artifact metadata that justified each dependency.
Run `scripts/verify_static.py` and add parser/behavior tests. A new GPU, Python
ABI, topology or native artifact needs a new profile and real acceptance—not an
edit that broadens an old claim.

## Contribution evidence

Include commands/results, source identity, environment facts and checksums.
Clearly label CPU-only, mocked, historical and newly executed evidence. Keep
large wheels, weights, archives, caches, staged runtimes and generated logs out
of Git. Preserve upstream vLLM attribution for upstream capabilities and keep
model licenses separate from SDK licensing.
