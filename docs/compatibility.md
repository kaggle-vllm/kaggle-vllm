# Compatibility boundary

| SDK | Validated Kaggle profile | Native runtime | Status |
| --- | --- | --- | --- |
| 0.2.x | 2 × NVIDIA Tesla T4, SM75, TP=2 | CPython 3.12, PyTorch 2.10.0+cu128, vLLM source v0.18.1 | Historical release only |
| 0.3.0 | Exactly 4 × NVIDIA L4, SM89, TP=4 | CPython 3.12/cp312, PyTorch 2.13.0+cu129, vLLM v0.30.0 | Accepted candidate wheel passed 19/19 cleanup-r1 gates; rebuilt SDK wheel retest required |

Local SDK development/build/test uses Python 3.11. It cannot import the cp312 native wheel. The packaged SDK declares `>=3.11,<3.13`; runtime activation requires CPython 3.12 on Linux x86_64.

The system CUDA compiler/toolkit in P31 was 12.8. PyTorch supplied the CUDA 12.9 user-space closure. The Python distribution `cuda_toolkit-12.9.1` does not replace `/usr/local/cuda`.

Strict hardware validation rejects fewer than four GPUs, T4, P100, A100, and L4 devices reporting a capability other than 8.9. P31 NCCL world_size=4 passed. No universal CPU-count requirement is encoded; the P31 source build derived parallelism from `nproc --all`, which returned 48 in that session.

---

## Preserved historical v0.2 documentation

The section below is retained verbatim from the v0.2 repository state for historical users. Commands and defaults above describe the v0.3 candidate.

# Compatibility contract

The current profile is `kaggle-t4x2-cu128`. “Validated” means observed in the
archived Kaggle sessions; it is not a general platform certification.

## Exact native requirements

| Dimension | Contract | Reason |
|---|---|---|
| Implementation/ABI | CPython 3.12 / cp312 | native wheel tag |
| OS/architecture | Linux x86_64 | wheel and ELF identity |
| PyTorch | 2.10.0+cu128 | compiled extension linkage |
| PyTorch CUDA ABI | 12.8 | compiled runtime boundary |
| GPU profile | 2 × Tesla T4, SM75 | strict T4x2 profile |
| NCCL | 2.27.5 | validated distributed runtime |

Bootstrap always rejects an incompatible Python ABI, OS or machine. With
`--strict`, all validated host dimensions become errors. The dependency doctor
also treats missing/out-of-range required distributions as errors; versions
inside upstream-supported ranges but different from the recorded overlay are
warnings normally and errors under `doctor --strict`.

## Observed but not universally fatal

- `flashinfer-python` is not required by the validated Tesla T4 / SM75
  `TRITON_ATTN` execution path. Upstream vLLM declares FlashInfer in its
  general CUDA requirements, but vLLM detects its absence at runtime.
  `kaggle-vllm doctor` therefore reports a missing FlashInfer installation as
  `UNTESTED` rather than `ERROR` for this profile.
- driver patch 580.159.04 and driver-reported CUDA maximum 13.0;
- CUDA toolkit patch 12.8.93, CMake 3.31.10 and GCC 11.4.0;
- exact versions of packages whose upstream constraints allow a range;
- `TRITON_ATTN` selection, FlashAttention 2 rejection on SM75 and non-fatal
  SymmMem warnings;
- compatibility-first eager/custom-all-reduce settings.

The driver-reported CUDA maximum is not GPU compute capability. For Tesla T4,
compute capability is 7.5 / SM75.

## Feature support matrix

| Feature | Status | Evidence boundary |
|---|---|---|
| Kaggle T4 x2 / cp312 / CUDA 12.8 | Validated | historical acceptance |
| Native imports (`_C`, `_moe_C`, allocator) | Validated | historical acceptance |
| OPT-125M TP=1 and TP=2 | Validated | functional plus controlled development-candidate benchmark |
| Qwen2.5-3B `sharded_state` TP=2 | Validated | exact rank topology |
| Local OpenAI models/completion/chat APIs | Validated | HTTP functional checks |
| SM75 FlashAttention 2 | Unavailable/not selected | `TRITON_ATTN` observed |
| TP1/TP2 performance matrix | Executed 2026-08-30 | OPT-125M only; TP=1 faster in recorded run |
| Python 3.10/3.11/3.13 lightweight SDK | CPU CI | native cp312 wheel excluded |
| Other Kaggle GPU architectures | Unvalidated | no profile/evidence |
| TP=1 or TP>2 for Qwen persistent state | Unsupported by profile | topology mismatch |
| Multi-node | Unvalidated | no evidence |
| Training/fine-tuning | Out of scope | inference delivery toolkit |
| Windows/macOS native runtime | Unsupported by profile | Linux wheel |

Machine-readable identities live in
[`compat/kaggle-t4x2-cu128.json`](../compat/kaggle-t4x2-cu128.json), the packaged
profile and [`artifacts/BUILD-PROVENANCE.json`](../artifacts/BUILD-PROVENANCE.json).
