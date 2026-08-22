# Validation plan

No production-ready installation is performed until all earlier gates pass.

## Gate 0 — artifact integrity

- verify wheel SHA256
- inspect wheel contents
- inspect `ldd` output
- inspect CUDA fatbins with `cuobjdump`

## Gate 1 — staged import only

Use `pip --target --no-deps` so normal Kaggle site-packages are not the install
target. Validate imports from the staged directory.

Expected imports:

- `vllm`
- `vllm._C`
- `vllm._moe_C`
- `vllm.cumem_allocator`

If Python dependency imports fail, stop and record the exact missing/incompatible
package before changing anything.

## Gate 2 — raw NCCL

Run `scripts/nccl_smoke.py`.

## Gate 3 — single T4 inference

Use FP16, `tensor_parallel_size=1`, conservative memory utilization, and
`enforce_eager=True`.

## Gate 4 — dual T4 tensor parallelism

Use FP16, `tensor_parallel_size=2`, `disable_custom_all_reduce=True` initially,
and conservative memory utilization.

## Gate 5 — serving

Only after offline generation works:
- start vLLM's OpenAI-compatible server
- test `/v1/models`
- test `/v1/chat/completions`

## Gate 6 — optimization

Only after correctness:
- benchmark eager vs CUDA graphs
- benchmark NCCL vs vLLM custom all-reduce on PHB topology
- remove irrelevant FA3/Hopper build targets if safe
- optimize wheel size
