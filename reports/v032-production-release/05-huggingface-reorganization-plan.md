# Additive Hugging Face reorganization

The Hub remains at `8d7dd28f7c76c3b68e7326df81758411b7dc2680` with 16 files. Historic T4 and L4 native paths remain available. No 0.3.2 namespace has been published.

The revised 0.3.2 design avoids mirroring third-party dependency wheels. All 168 exact locked files were found at their publisher indexes: 162 on PyPI, five on the PyTorch CUDA 12.9 index, and one on FlashInfer. The publisher source inventory is in `upstream-wheel-origins.json`; 167 hashes were advertised by the indexes, and the remaining TorchVision wheel was independently downloaded and SHA256 verified. The SDK still verifies every downloaded file by exact size and SHA256 before staging.

The proposed additive Hub change has 14 metadata/index additions in `release/hf-v032-upload-plan.json`, including `UPSTREAM_WHEEL_SOURCES.json`. It copies **no native wheels** and uploads **zero dependency wheels**. The SDK uses the exact native wheel already present at `v0.3.0/l4x4/native/`, pinned by the final immutable revision and SHA256. The `v0.3.2/l4x4/wheelhouse/` path stays absent by design; dependency bytes come from publishers.

Before any external mutation, document the native wheel's compiled-component redistribution rights and notices, verify authorized account credentials, and confirm the Hub base revision still matches the plan. The user has authorized publication contingent on those prerequisites. After publication, verify every target size/hash and the immutable Hub commit, then pin it in the SDK and rebuild. Keep old paths.

The initial full-mirror plan and its 175 missing target paths remain historical evidence in `missing-public-l4-artifacts.json`. The revised plan supersedes its dependency-wheel uploads and native copies. Reusing an already public binary avoids another redistribution, but it does not settle the original binary provenance and notice requirements.
