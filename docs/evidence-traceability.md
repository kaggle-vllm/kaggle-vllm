# Evidence traceability

This map connects public claims to the smallest authoritative evidence and the
maintained repository explanation. Paths beginning `Kaggle-Session-Files-3/`
refer to the maintainer's read-only full evidence collection; Git-safe excerpts
are under `artifacts/kaggle-2026-08-23/`.

| Claim | Primary evidence | Repository surface | Status |
|---|---|---|---|
| Python 3.12.13, Torch 2.10.0+cu128, toolkit 12.8.93, driver 580.159.04, T4x2 SM75, CMake/GCC | `kaggle-build-environment.txt`, evidence `runtime-fingerprint.txt` | `compat/kaggle-t4x2-cu128.json`, [compatibility](compatibility.md) | Historically validated |
| Driver reports CUDA maximum 13.0; GPU compute capability is 7.5 | `kaggle-build-environment.txt` | [Kaggle runtime](kaggle-runtime.md), environment fields | Observed; terminology corrected |
| Native wheel SHA256 and immutable revision | local `sha256sum`, `KAGGLE-VLLM-DOWNLOAD-SHA256SUMS.txt`, public HF revision | packaged profile, [provenance](provenance.md) | Independently verified |
| Source is vLLM v0.18.1 at `a26e8dc…`; wheel has generated 0.18.2.dev0 version | upstream Git tag object, `vllm-source-identity.txt`, wheel `METADATA` | [native runtime](native-runtime.md) | Verified; identities distinguished |
| ELF x86-64/cp312 structure and SM75 target | wheel `WHEEL`/`RECORD`, `readelf`, `cuobjdump`, evidence `vllm-wheel-cuda-architectures.txt` | [native runtime](native-runtime.md) | Structurally verified |
| Native modules import from staged runtime | `vllm-native-imports.log`, executed v0.1.1/v0.1.2 notebooks | [validation](validation.md) | Historically validated |
| Raw two-rank NCCL works | `kaggle-dual-t4-nccl-smoke.log` | [multi-GPU](multi-gpu.md) | Historically validated |
| OPT-125M single T4 and TP=2 generate | single/dual scripts and recovery logs, evidence `result.json` | [validation](validation.md) | Historically functional; not benchmarked |
| SM75 selects `TRITON_ATTN`; FA2/SymmMem limitations are non-fatal | OPT/Qwen/server logs | [multi-GPU](multi-gpu.md), [troubleshooting](troubleshooting.md) | Observed for exact profile |
| Qwen persistent state is ranks 0–1, two parts each, 6,172,262,512 bytes | archive listing, shard manifest, per-shard checksum manifest, reload log | [sharded state](sharded-state.md) | TP=2 validated; other topology unsupported |
| Qwen archive hash and license boundary | local archive SHA256, archive `LICENSE`, public Qwen license | [provenance](provenance.md) | Verified; historical archive lacks later NOTICE |
| Local OpenAI models/completion/chat endpoints return success | recorded server logs and response JSON | [OpenAI serving](openai-serving.md) | Historically functional only |
| Manifest-owned reset preserves cache and re-bootstraps | executed v0.1.2 notebook | [v0.1.2 acceptance](kaggle-v0.1.2-reset-acceptance.md) | Historically validated |
| Dependency baseline matches the known overlay | wheel `METADATA`, upstream requirements, overlay resolution/install logs | [doctor](doctor.md), packaged baseline | Locally tested; new strict run pending Kaggle |
| TP1 versus TP2 performance | no executed 0.2.0 result | [benchmarking](benchmarking.md), pending notebook | Pending; no performance claim |

Generated text in evidence demonstrates that inference completed. It is not a
quality evaluation. CPU unit tests validate SDK behavior but do not replace any
GPU evidence row.
