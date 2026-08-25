---
pretty_name: vLLM Kaggle validated binaries
tags:
- vllm
- kaggle
- cuda
- tesla-t4
---

# vLLM Kaggle validated binaries

This repository distributes a vLLM wheel built and functionally validated on a
Kaggle Notebook with two NVIDIA Tesla T4 (SM75) GPUs. It is packaging work
around upstream [vLLM](https://github.com/vllm-project/vllm), not a fork or a
claim of ownership over vLLM. **This is not an official upstream vLLM binary.**

The repository also carries exact PyPI artifacts for the lightweight
`kaggle-vllm` 0.1.2 SDK, plus the historical 0.1.0 and 0.1.1 SDK files. These
pure-Python artifacts do not contain vLLM, CUDA, or Torch and do not install
the native runtime implicitly.

## Lightweight SDK artifacts

| File | SHA256 |
|---|---|
| `kaggle_vllm-0.1.2-py3-none-any.whl` | `13f1043df4a173e74555c6a4d7a8f66b4e661d942fc0222124208f50b1e9aad2` |
| `kaggle_vllm-0.1.2.tar.gz` | `f05c210985bcc74fad689a1ffaf0c4200e80041e791d902e1703b279db02679e` |
| `kaggle_vllm-0.1.1-py3-none-any.whl` | `d8dfb58e369ceea90b2ade10c75d7678166615a04cbea120855bfd2329bbc9db` |
| `kaggle_vllm-0.1.1.tar.gz` | `c9981a564513b596bdbd0a68365230d2eb330a61b6b28e42fc22c043b5169349` |
| `kaggle_vllm-0.1.0-py3-none-any.whl` | `e6b525d03257f24e2e062770763bf060042fe4868f879fb6f81efc722b076233` |
| `kaggle_vllm-0.1.0.tar.gz` | `a35776573291846f20747dad17e193bd00b4dbb4853294224f17db045c51dd0a` |

The primary installation source is PyPI:

```bash
pip install kaggle-vllm==0.1.2
kaggle-vllm bootstrap
```

The immutable Hub fallback uses the byte-identical PyPI wheel:

```bash
pip install "https://huggingface.co/waqasm86/kaggle-vllm-binaries/resolve/97b741d7fc988ed557a00fc28f2e34abad09fb7d/kaggle_vllm-0.1.2-py3-none-any.whl#sha256=13f1043df4a173e74555c6a4d7a8f66b4e661d942fc0222124208f50b1e9aad2"
kaggle-vllm bootstrap
```

The historical 0.1.1 fallback remains available at revision
`ff213d775c560645dbd1bdaf86f7412005717969` with its recorded wheel checksum.

The historical 0.1.0 fallback remains pinned to its original publication
commit:

```bash
pip install "https://huggingface.co/waqasm86/kaggle-vllm-binaries/resolve/ec75826d10e2dbc3c94c4682342ea3b65d7b72e2/kaggle_vllm-0.1.0-py3-none-any.whl#sha256=e6b525d03257f24e2e062770763bf060042fe4868f879fb6f81efc722b076233"
kaggle-vllm bootstrap
```

The bootstrap profile remains pinned to the native wheel revision and checksum;
it does not use mutable `main` for native delivery.

## Binary identity

- Source: upstream vLLM v0.18.1
- Commit: `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`
- Wheel version: `0.18.2.dev0+ga26e8dc7f.d20260822.cu128`
- ABI: CPython 3.12, Linux x86_64
- SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

The differing source and wheel version strings result from `setuptools_scm`
metadata generation; the source checkout was the v0.18.1 tag.

## Validated runtime

Python 3.12.13, PyTorch 2.10.0+cu128, CUDA toolkit 12.8.93, driver
580.159.04, NCCL 2.27.5, and two Tesla T4 GPUs (SM75). Validation covered
native imports, single-GPU inference, NCCL, TP=2 inference, Qwen2.5-3B FP16,
vLLM `sharded_state` persistence/reload, and OpenAI-compatible serving.

FlashAttention 2 is unavailable on the Tesla T4's SM75 architecture. During
the validated runs, vLLM selected `TRITON_ATTN` successfully. SymmMem
capability warnings are expected on SM75; ordinary NCCL tensor-parallel
communication still worked.

Use the supplied checksums and compatibility JSON before staging. Avoid normal
dependency resolution that could replace Kaggle's Torch stack; the associated
[`kaggle-vllm`](https://github.com/kaggle-vllm/kaggle-vllm)
project documents explicit, checksum-verified `pip --target --no-deps`
bootstrap staging.

The wheel includes upstream vLLM's Apache-2.0 license. Compatibility beyond the
documented environment is not claimed. In particular, this artifact is not a
claim of universal CUDA, Python, PyTorch, GPU, or platform compatibility.
