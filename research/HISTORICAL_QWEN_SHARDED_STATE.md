# Historical Qwen TP-aware sharded-state evidence

Status: **VERIFIED HISTORICAL CAPABILITY EVIDENCE; NOT AN M4 INPUT**.

The external archive
`Kaggle-Session-Files-3/qwen2.5-3b-t4x2-sharded.tar.gz` is 4,773,220,584 bytes
and has SHA256
`12dcb264cb74e6fa2947b5f1fbebfa14562afa2292387f49e447b3290bc0b83b`.
On 2026-09-11, streaming `sha256sum -c` verification against its sidecar
passed. The archive was listed without extraction and contains the same 14
files and sizes as `qwen3b-sharded-state-manifest.txt`: four TP-rank
safetensors parts totaling 6,172,262,512 bytes plus pinned Qwen tokenizer,
configuration, license, notice, and model-index metadata. Total manifested
uncompressed content is 6,183,800,455 bytes.

The associated save log ends in `SHARDED STATE SAVE PASS`. The reload log uses
vLLM `load_format=sharded_state`, TP=2, and FP16, loads both ranks, generates
two responses, and ends in `SHARDED STATE TP=2 RELOAD PASS`. The OpenAI server
log records readiness from the sharded-state path; the retained `/v1/models`
and chat-completion JSON show a successful served response with token usage.
The source identity is upstream vLLM tag `v0.18.1`, commit
`a26e8dc7ff2111a005144d775ecf9cebf56c45b2`; the native wheel identity remains
the separately recorded project wheel SHA256. The 382,096-byte RC2 evidence
bundle and its sidecar also verify and retain the save/reload scripts, logs,
manifest, responses, runtime records, and the large archive's checksum without
duplicating the model archive.

This supports only the statement that earlier kaggle-vllm acceptance evidence
demonstrated persistence and reload of a TP=2 Qwen2.5-3B vLLM sharded state on
the same dual-T4 runtime. It does not establish a novel checkpoint format,
does not imply identical support across architectures, and does not make the
large archive a prerequisite for M4.

M4 starts each cell from the regular pinned Hugging Face checkpoint with
`model_source=huggingface` and `model_representation=transformers`; it never
uses `load_format=sharded_state`. Creating Phi, Ministral, or Llama archives
now would therefore not improve M4's internal validity. Creating any of the
following archives is classified **OUT_OF_SCOPE_FOR_PRIMARY_M4**:

- Phi-4 Mini: no principal-question value; substantial GPU/storage cost and a
  new persistence variable and provenance obligation.
- Ministral 3 3B BF16: the same constraints, plus model-card and
  redistribution review before artifact creation.
- Llama 3.2 3B: the same constraints, with mandatory gated-access, license,
  and redistribution review.
- Gemma 3 4B: ineligible because the frozen SM75 protocol never reached
  serving readiness; no persistence artifact should be generated.

Creating archives only for models observed to pass compatibility could also
appear post-hoc if presented as part of the original performance study. No
archive above should be created merely to mirror the historical Qwen file.

## Future work: cross-architecture persistence study

A separate future study may ask whether kaggle-vllm can create, persist,
redistribute where legally permitted, reload, and validate TP-aware vLLM
sharded states consistently for Qwen, Phi, Ministral, and Llama on dual T4.
Before execution it must freeze a license and redistribution review, model-card
review, artifact-size estimate, storage destination, checksum scheme, topology
metadata, and save/load/reload benchmark protocol. Gemma remains excluded
under the frozen SM75 study. This document authorizes no artifact generation or
upload.
