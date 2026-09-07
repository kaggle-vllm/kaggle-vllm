# TP=2 model-artifact workflow

Every candidate runs in a separate fresh Kaggle T4 x2 session. The generic
runner reads only a reviewed entry in `model_matrix.json`; unknown models and
changed model-card license identifiers fail closed.

1. Strictly validate kaggle-vllm 0.2.0, the exact native wheel hash, vLLM
   distribution/source identity, Python/Torch/CUDA/NCCL/driver, two T4 SM75
   devices and PHB/P2P topology.
2. Resolve the pinned Hugging Face revision and model-card license. Retrieve
   `HF_TOKEN` only through Kaggle Secrets; never print it. Access is not
   redistribution permission.
3. Check projected decimal 20 GB working-storage use. Monitor download, TP2
   save, fresh reload and API phases for the 28 GiB RAM and 14.5 GiB/GPU caps.
4. Download only reviewed allow-listed assets. Hash every selected source file
   and require the reviewed weight-byte total. Duplicate consolidated Mistral
   weights are excluded.
5. Load real FP16 TP=2 with the validated SM75 path, prefix caching off and
   ordinary NCCL. Call the exact upstream
   `llm.llm_engine.engine_core.save_sharded_state` API with a 2 GiB part limit.
6. Inspect contiguous rank/part topology and hash every shard. Terminate the
   save process. Start a fresh offline process and load only `sharded_state`;
   run deterministic generation.
7. Start a new OpenAI-compatible server from only the saved state. Preserve
   real `/v1/models` and `/v1/completions` responses and prove rank 0, rank 1
   and `TRITON_ATTN` in logs.
8. Remove the original source/cache, then create and stream-hash the tarball.
   Verify archive members without extracting a second copy. Preserve a
   manifest, validation, provenance, checksum and license-attribution record.
9. Upload only when `redistribution_status` explicitly permits it. Verify every
   remote object and exact size. Only then delete the local tar, sharded state
   and caches; collect garbage, empty the CUDA cache and record reclaimed
   resource state. Llama/Gemma stay `UPLOAD_BLOCKED_LICENSE_REVIEW`.

The result is a topology-specific vLLM-native TP=2 sharded state. It is not a
newly trained/fine-tuned model, ordinary Transformers checkpoint, or
topology-independent artifact. Preserve unsupported/OOM runs as negative
evidence.
