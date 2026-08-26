# Next Kaggle T4x2 acceptance

Use a fresh Kaggle notebook with two Tesla T4 GPUs. The pending
[`kaggle_vllm_0_2_0_acceptance.ipynb`](../kaggle-notebooks/kaggle_vllm_0_2_0_acceptance.ipynb)
is the starting artifact; replace its SDK SHA placeholder with the reviewed
release-candidate wheel digest before running.

## Required sequence

1. Capture `python --version`, glibc, `nvidia-smi`, `nvidia-smi topo -m`,
   `nvcc --version`, CMake and GCC.
2. Record Torch version, CUDA ABI and installation path before SDK install.
3. Verify the small SDK wheel SHA256 and install it; confirm no Torch/vLLM/CUDA
   packages were pulled.
4. Run fingerprint and strict bootstrap dry-run; verify native filename,
   immutable revision and SHA256.
5. Bootstrap, activate the manifest and run `doctor --strict --json` so the
   dependency baseline—including xgrammar/tvm_ffi—passes.
6. Import `vllm`, `vllm._C`, `vllm._moe_C` and
   `vllm.cumem_allocator`; verify Torch version/path remain identical.
7. Run `scripts/nccl_smoke.py`.
8. Run a focused OPT-125M TP=1 smoke, fully release it, then TP=2.
9. If Qwen coverage is required, avoid redownloading when an immutable attached
   dataset is available; verify its identity, inspect with expected TP=2,
   reload with `load_format=sharded_state`, and generate.
10. Launch the server on `127.0.0.1`, verify models plus completion/chat as
    appropriate, and terminate it cleanly.
11. Execute the benchmark notebook one configuration per process and retain
    every result, including failures.
12. Save outputs under a new dated 0.2.0 evidence name, checksum them, terminate
    all workers and confirm a clean exit.

## Acceptance boundary

Do not mark benchmark, new TP1/TP2 comparison or new 0.2.0 acceptance as passed
until the executed notebook and JSON evidence are reviewed. Historical 0.1.x
functional evidence remains valid historical evidence but is not a substitute
for testing new code paths.
