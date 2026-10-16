# Additive Hugging Face reorganization

The Hub remains at `8d7dd28f7c76c3b68e7326df81758411b7dc2680` with 16 files. Historic T4 and L4 native paths remain available. No 0.3.2 namespace has been published.

The revised 0.3.2 design avoids mirroring third-party dependency wheels. All 168 exact locked files were found at their publisher indexes: 162 on PyPI, five on the PyTorch CUDA 12.9 index, and one on FlashInfer. The publisher source inventory is in `upstream-wheel-origins.json`; 167 hashes were advertised by the indexes, and the remaining TorchVision wheel was independently downloaded and SHA256 verified. The SDK still verifies every downloaded file by exact size and SHA256 before staging.

The proposed additive Hub change has 16 operations in `release/hf-v032-upload-plan.json`: copy the two existing LFS native wheels into versioned paths and add the staged release index and metadata, including `UPSTREAM_WHEEL_SOURCES.json`. It uploads **zero dependency wheels**. The `v0.3.2/l4x4/wheelhouse/` path stays absent by design; the versioned namespace supplies native and metadata while dependency bytes come from publishers.

Before any external mutation, document the native wheel's compiled-component redistribution rights and notices, obtain the user's explicit approval for the exact additive commit, and confirm the Hub base revision still matches the plan. After publication, verify every target size/hash and the immutable Hub commit, then pin it in the SDK and rebuild. Keep old paths; deletion requires separate approval.

The initial full-mirror plan and its 175 missing target paths remain historical evidence in `missing-public-l4-artifacts.json`. The revised plan supersedes its dependency-wheel uploads. [Hugging Face documents server-side LFS copying](https://huggingface.co/docs/huggingface_hub/guides/repository).
