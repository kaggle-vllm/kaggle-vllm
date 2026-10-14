# Next Kaggle acceptance: corrected split-v2 Inputs

Use the [fresh split-v2 acceptance guide](fresh-kaggle-v030-acceptance.md), [run-first instructions](../release/README-v0.3.0-production-acceptance.md), and [output-free notebook](../kaggle-notebooks/kaggle_vllm_0_3_0_l4x4_final_sdk_acceptance.ipynb). The current acceptance reuses four corrected v2 Datasets and the saved P31 Notebook Output. It does **not** require a manual 6.6 GB runtime-v3 Dataset upload. The small SDK package contains no runtime dependency wheels.

The runner verifies exact CPython 3.12/four-L4/SM89 hardware, hashes the SDK and every selected runtime/native wheel, stages the cu129 overlay offline, checks doctor/native core/FA2 import and libcuda linking, runs NCCL world_size=4, then tests the public `KaggleLLM.chat()` Gemma TP4 semantic path and worker cleanup. Its bounded child processes stream saved logs. Download the final evidence ZIP and inspect `FRESH_KAGGLE_PASS`.

The supported competition model is `gemma-4-31b-it-qat-w4a16-ct`. This notebook does not run Qwen or another unrelated model on restricted competition L4 capacity. P31 Gemma selected `TRITON_ATTN`; built-in FA2 binary/import is checked, while actual FA2-backed model inference remains unproven for a later permitted environment.

---

## Preserved historical v0.2 documentation

The section below is retained verbatim from the v0.2 repository state for historical users. Commands and defaults above describe the v0.3 candidate.

# Final Kaggle T4x2 acceptance procedure and result

The SDK is public on PyPI. This procedure was executed successfully on a fresh
Kaggle notebook with two Tesla T4 GPUs on 2026-08-31. The executed
[`kaggle_vllm_0_2_0_post_publication_acceptance.ipynb`](../kaggle-notebooks/kaggle_vllm_0_2_0_post_publication_acceptance.ipynb)
installs exactly `kaggle-vllm[hub]==0.2.0`. The already-passed development
candidate remains separate evidence. See the reviewed
[final acceptance report](kaggle-v0.2.0-final-acceptance.md).

## Required sequence

1. Capture `python --version`, glibc, `nvidia-smi`, `nvidia-smi topo -m`,
   `nvcc --version`, CMake and GCC.
2. Record Torch version, CUDA ABI and installation path before SDK install.
3. Install the exact public `kaggle-vllm[hub]==0.2.0` package, record installed
   distribution metadata and confirm no Torch/vLLM/CUDA packages were pulled
   as SDK dependencies.
4. Run fingerprint and strict bootstrap dry-run; verify native filename,
   immutable revision and SHA256.
5. Bootstrap, activate the manifest and run `doctor --strict --json` so the
   dependency baseline—including xgrammar/tvm_ffi—passes.
6. Import `vllm`, `vllm._C`, `vllm._moe_C` and
   `vllm.cumem_allocator`; verify Torch version/path remain identical.
7. Run `scripts/nccl_smoke.py`.
8. Run a focused OPT-125M TP=1 smoke, fully release it, then TP=2.
9. Run the separate focused Qwen notebook if release scope requires it. Prefer
   the already-published artifact through an immutable attached dataset or
   cache, inspect with expected TP=2, verify symlink/topology safeguards, reload
   with `load_format=sharded_state`, and generate without regenerating shards.
10. Launch the server on `127.0.0.1`, verify models plus completion/chat as
    appropriate, and terminate it cleanly.
11. The controlled development-candidate benchmark need not be repeated unless
    final source or dependencies changed in a way that can affect performance.
12. Save outputs under a new dated 0.2.0 evidence name, checksum them, terminate
    all workers and confirm a clean exit.

## Recorded result

The exact public-package acceptance and focused Qwen regression are `PASS`.
Their executed notebooks and small machine-readable evidence were reviewed and
checksummed. The 2026-08-30 development-candidate acceptance/benchmark and
historical 0.1.x evidence remain valid for their exact identities and are not
rewritten by the final run.
