# Next Kaggle acceptance: published SDK 0.3.1

The SDK 0.3.1 wheel and source distribution are already published. Run the [offline Input audit](../kaggle-notebooks/gemma4-competition/01_offline_inputs_audit.ipynb) and [fresh SDK acceptance](../kaggle-notebooks/gemma4-competition/02_sdk_native_runtime_offline.ipynb) in separate private, Internet-disabled four-L4 sessions attached to the Gemma 4 Developer Agent competition. Attach the datasets and licensed model listed in the [notebook README](../kaggle-notebooks/gemma4-competition/README.md), including the corrected wheelhouse audit Input and private Python 3.12 supplemental base Input. These Inputs reuse the exact 168-wheel P33 dependency closure and unchanged native vLLM 0.30.0 wheel; no new CUDA build or public artifact upload is required.

The acceptance runner verifies CPython 3.12, four L4/SM89 GPUs, exact SDK and runtime wheel SHA256 identities, PyTorch 2.13.0+cu129, native vLLM 0.30.0, NCCL TP4, Gemma generation, backend selection, and worker cleanup. Download and inspect the final evidence bundle, checksum manifest, SDK import origin, and 19-gate matrix before recording a pass. Earlier 0.3.0 native-wheel acceptance remains evidence for the unchanged native bytes, but does not establish 0.3.1 SDK integration.

The research notebooks use `gemma-4-31b-it-qat-w4a16-ct`. The organizer's current harness also permits 12B, E4B, and E2B quantized variants, but every agent and subagent in one submission must use the same base variant. P31 Gemma selected `TRITON_ATTN`; built-in FA2 binary/import and Gemma backend selection are distinct checks.

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
