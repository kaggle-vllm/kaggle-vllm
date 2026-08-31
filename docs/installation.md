# Installation, bootstrap, and recovery

## Three SDK installation paths

The distribution name is `kaggle-vllm` and the import is `kaggle_vllm`.
Python packaging treats hyphens and underscores equivalently. The recommended
Kaggle flow installs the optional Hub client and applies strict profile checks:

```bash
pip install "kaggle-vllm[hub]==0.2.0"
kaggle-vllm bootstrap --strict
```

The equivalent normalized project spelling is:

```bash
pip install "kaggle_vllm[hub]==0.2.0"
```

The canonical distribution spelling resolves to the same normalized project:

```bash
python -m pip install "kaggle-vllm==0.2.0"
```

The exact public 0.2.0 PyPI files are recorded in the
[SDK checksum manifest](../release/kaggle-vllm-sdk-SHA256SUMS.txt). The binary
repository's latest mirrored SDK fallback remains the historical 0.1.2 wheel,
pinned to its immutable Hub commit and checksum; use PyPI for 0.2.0:

```bash
pip install "https://huggingface.co/waqasm86/kaggle-vllm-binaries/resolve/97b741d7fc988ed557a00fc28f2e34abad09fb7d/kaggle_vllm-0.1.2-py3-none-any.whl#sha256=13f1043df4a173e74555c6a4d7a8f66b4e661d942fc0222124208f50b1e9aad2"
kaggle-vllm bootstrap
```

The historical 0.1.1 fallback remains pinned to its original publication
commit and wheel checksum:

```bash
pip install "https://huggingface.co/waqasm86/kaggle-vllm-binaries/resolve/ff213d775c560645dbd1bdaf86f7412005717969/kaggle_vllm-0.1.1-py3-none-any.whl#sha256=d8dfb58e369ceea90b2ade10c75d7678166615a04cbea120855bfd2329bbc9db"
```

The historical 0.1.0 fallback remains available at its original immutable
publication commit and wheel checksum:

```bash
pip install "https://huggingface.co/waqasm86/kaggle-vllm-binaries/resolve/ec75826d10e2dbc3c94c4682342ea3b65d7b72e2/kaggle_vllm-0.1.0-py3-none-any.whl#sha256=e6b525d03257f24e2e062770763bf060042fe4868f879fb6f81efc722b076233"
kaggle-vllm bootstrap
```

The public 0.2.0 command was verified in a fresh Python 3.11 virtual
environment. The base package installs the SDK plus the small `packaging`
library used for PEP 440 checks; the `hub` extra adds the Hugging Face client.
Bootstrap separately obtains the cp312 native wheel. Neither installation path
installs vLLM, Torch or CUDA packages.

The SDK is pure Python and supports local development on Python 3.11. The
native artifact is a Linux x86_64 CPython 3.12 wheel. Bootstrap always rejects
Python 3.11 for that `cp312` profile; it never tries to install an incompatible
wheel.

## Why staging exists

Kaggle ships a coordinated PyTorch/CUDA/Triton environment. A normal vLLM pip
install can resolve a large dependency graph and replace packages in that
environment. The validated recovery kept PyTorch 2.10.0+cu128 at its system
path, staged only the wheel with `--no-deps`, and created a separate pinned
overlay for missing Python dependencies.

Installing the SDK does not download vLLM. Importing the SDK has no network or
installation side effects. Native-runtime work happens only when the user
explicitly runs `kaggle-vllm bootstrap` or calls the Python `bootstrap()` API.

## The `kaggle-t4x2-cu128` profile

The packaged profile records CPython 3.12.13/cp312, PyTorch 2.10.0+cu128, CUDA
toolkit 12.8.93, two Tesla T4 GPUs at SM75, NCCL 2.27.5, upstream vLLM v0.18.1
at `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`, and the exact overlay lock.
Its native wheel source is pinned to:

```text
repository: waqasm86/kaggle-vllm-binaries
revision:   f6b4f10de54924ed6fe9e28cceab84eca7276ab6
wheel:      vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl
SHA256:     5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c
```

When `huggingface_hub` is installed, bootstrap uses `hf_hub_download()` and
normal Hub cache/Xet delivery. Otherwise it follows the official HTTPS
`/resolve/` redirect with the standard library. Both routes use the immutable
revision and verify the wheel before staging.

## Dry run and strict validation

```bash
kaggle-vllm bootstrap --dry-run --strict
```

Dry run reports the profile, host findings, artifact identity, cache and target
paths, and exact pip argument arrays. It neither creates directories nor
downloads anything. Linux, x86_64, CPython, and cp312 are always mandatory for
the current native wheel. Without `--strict`, Kaggle/Torch/CUDA/GPU/NCCL drift
is reported as warnings; strict mode makes every validated-profile mismatch an
error.

After bootstrap activation, validate both the host and installed dependency
baseline:

```bash
eval "$(kaggle-vllm env)"
kaggle-vllm doctor --strict
kaggle-vllm doctor --json > /kaggle/working/kaggle-vllm-doctor.json
```

The dependency baseline is derived from native wheel metadata and the validated
overlay. Run it after activation; before activation, missing overlay packages
are expected errors.

The defaults can be replaced explicitly:

```bash
kaggle-vllm bootstrap --strict \
  --cache /kaggle/working/kaggle-vllm-cache \
  --staged /kaggle/working/vllm-staged \
  --overlay /kaggle/working/vllm-runtime-overlay \
  --manifest /kaggle/working/kaggle-vllm-runtime.json
```

The 2026-08-25 acceptance run used notebook-owned destinations and a reusable
download cache. This avoids collisions with a prior runtime while preserving
bootstrap's safety checks:

```bash
RUNTIME_ROOT=/kaggle/working/kaggle-vllm-e2e-011

kaggle-vllm bootstrap --strict \
  --cache /kaggle/working/kaggle-vllm-cache \
  --staged "$RUNTIME_ROOT/vllm-staged" \
  --overlay "$RUNTIME_ROOT/vllm-runtime-overlay" \
  --manifest "$RUNTIME_ROOT/kaggle-vllm-runtime.json"

export KAGGLE_VLLM_MANIFEST="$RUNTIME_ROOT/kaggle-vllm-runtime.json"
eval "$(kaggle-vllm env --manifest "$KAGGLE_VLLM_MANIFEST")"
```

The staged and overlay destinations must be absent or empty. Refusing to
overwrite a non-empty destination protects user data and prevents accidental
mixing of incompatible runtimes. Choose a new notebook-owned root for a clean
acceptance run; do not delete an existing directory unless you have inspected
and identified it yourself.

In that measured session, the native bootstrap/runtime occupied approximately
2.2 GiB. The four Qwen rank-specific weight files totaled 6,172,262,512 bytes
(about 5.75 GiB), and the Kaggle UI showed about 8 GiB of total working/output
usage after the full Qwen acceptance flow. These are reference observations,
not fixed requirements: caches, package metadata, and filesystem accounting can
change the total.

## Recovering from an existing staged runtime

The default refusal to overwrite a non-empty staged or overlay directory is a
safety feature. Version 0.1.2 adds an explicit reset plan for SDK-owned runtime
state without weakening that default.

Review the plan first:

```bash
kaggle-vllm bootstrap --reset-runtime --dry-run --strict
```

Add `--json` for a machine-readable plan. It reports the selected staged,
overlay, manifest, and cache paths; whether each exists and is owned; and
whether it would be removed, preserved, or skipped as absent.

After reviewing the plan, explicitly confirm removal and continue directly
into the normal bootstrap:

```bash
kaggle-vllm bootstrap --reset-runtime --yes --strict
```

Only the selected staged directory, dependency overlay, and runtime manifest
are removed. The download cache is preserved by default. Custom existing paths
must be recorded by the selected manifest, and that manifest must exactly match
all selected CLI paths. Any mismatch stops the operation rather than guessing.

Reset refuses root and system paths, `/tmp`, `/kaggle`, `/kaggle/input`,
`/kaggle/working`, the home or current directory, repository contents,
overlapping parent targets, and paths that traverse symlinks. It never prompts
users to run broad commands such as `rm -rf /kaggle/working/*`. This behavior
implements the safety requirements tracked in
[issue #7](https://github.com/kaggle-vllm/kaggle-vllm/issues/7).

## Wheel integrity and staging

The validated binary, checksum files, compatibility manifest, and provenance
metadata are published at
[waqasm86/kaggle-vllm-binaries](https://huggingface.co/waqasm86/kaggle-vllm-binaries).
It is project packaging derived from upstream vLLM, not an official upstream
vLLM wheel or a universal compatibility claim.

```bash
DIGEST=5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c
kaggle-vllm verify-wheel /kaggle/input/.../vllm-*.whl --sha256 "$DIGEST"
kaggle-vllm stage-wheel /kaggle/input/.../vllm-*.whl \
  --target /kaggle/working/vllm-staged \
  --sha256 "$DIGEST"
```

The destination must be absent or empty. The tool does not delete an existing
environment and does not perform a global install.

## Dependency overlay

The archived `vllm-overlay-lock-v3.txt` is the reproducible input from the
successful recovery. Python callers can use `stage_dependency_overlay()`; it
rejects `torch`, `torchvision`, and `torchaudio` requirement entries, uses
`pip --target --no-deps`, and requires an empty target.

The runtime ordering proven in the notebook was:

```text
PYTHONPATH=/kaggle/working/vllm-runtime-overlay:/kaggle/working/vllm-staged
```

`LD_LIBRARY_PATH` included the existing Torch library directory, the mounted
driver directory, and the CUDA toolkit library directory. Validate native
imports before running inference.

Successful bootstrap writes a small JSON runtime manifest. `KaggleLLM` can
activate that completed manifest if its first lazy vLLM import fails. For an
interactive shell or the upstream server CLI, activate it explicitly without
editing `.bashrc`:

```bash
eval "$(kaggle-vllm env)"
```

This prepends the overlay and staged directories to `PYTHONPATH`, the staged
console-script directory to `PATH`, and the existing Torch/NVIDIA/CUDA library
directories to `LD_LIBRARY_PATH`.

## Persistent Qwen TP=2 model

The published model is a topology-aware vLLM `sharded_state`, not a normal
Transformers checkpoint. To make repository resolution explicit, download a
snapshot through the normal Hub API and pass its local path:

```python
from huggingface_hub import snapshot_download
from kaggle_vllm import KaggleLLM

model_path = snapshot_download("waqasm86/kaggle-vllm-models")
llm = KaggleLLM(
    model=model_path,
    tensor_parallel_size=2,
    load_format="sharded_state",
)
```

It was validated only at TP=2. TP=1, TP>2, uneven splitting, arbitrary tensor
splitting, and standard Transformers loading are not supported claims.

## Source build

`scripts/prepare_vllm_0181.sh` checks out the exact upstream tag/commit and runs
vLLM's `use_existing_torch.py --prefix`. `scripts/build_vllm_0181.sh` selects
CUDA/SM75, applies the driver discovery path, avoids dependency wheel building,
and records a full log. These scripts are Kaggle-oriented and are not invoked by
SDK import.
