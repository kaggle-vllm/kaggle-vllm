# Doctor and dependency diagnostics

`doctor` compares the current process with the exact Kaggle T4x2 profile and a
curated subset of critical native-runtime distributions:

```bash
kaggle-vllm doctor
kaggle-vllm doctor --strict
kaggle-vllm doctor --json
```

The baseline is packaged at
`src/kaggle_vllm/profiles/kaggle-t4x2-cu128/dependency-baseline.json`. It was
derived from the native wheel `METADATA`, upstream v0.18.1 requirements and the
validated recovery overlay—not from an arbitrary package list. It covers core
model/tokenizer, API, serialization, grammar and backend packages, including
the historical `xgrammar==0.2.3` / `apache-tvm-ffi==0.1.13.post3` pair.

## Status meanings

| Status | Meaning |
|---|---|
| `PASS` | installed version matches its requirement and recorded exact version, if any |
| `WARNING` | supported range matches, but version differs from the validated overlay |
| `ERROR` | required package is missing, version is malformed/out of range, or strict drift occurred |
| `UNTESTED` | optional/informational dependency is absent |

An `ERROR` or host-profile mismatch returns exit code 1. Warnings and untested
optional packages remain visible without automatically declaring the runtime
unusable. `--strict` promotes exact-version drift to errors.

Run doctor after activating the completed overlay. Before bootstrap, missing
overlay packages are expected to appear as errors. `--no-dependencies` is
available for host-only diagnosis, not as a substitute for final acceptance.

JSON output contains the secret-free environment, profile findings, dependency
findings, status counts and final compatibility boolean for evidence capture.
