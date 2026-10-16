# L4 runtime wheelhouse integrity

The P33 inference lock has 168 unique dependency wheels totaling 6,399,476,597 bytes, plus the 293,855,397-byte native wheel. The historical local verifier reported 170 PASS and 0 non-PASS across the dependencies and both native wheels. This turn also confirmed all 168 local paths and sizes in the upload plan.

The revised release candidate retrieves dependencies from their original publishers instead of mirroring them on Hugging Face. `upstream-wheel-origins.json` records 168 exact matches: 162 PyPI, five PyTorch CUDA 12.9, and one FlashInfer. For 167 files the publisher index advertised the locked SHA256. The 93 MB TorchVision file had no index hash fragment, so its bytes were downloaded and SHA256 verified. PyPI's Triton 3.7.1 file has a different SHA256 under the same filename; only the matching PyTorch-index file is allowlisted. The packaged `UPSTREAM_WHEEL_SOURCES.json` is checksum-pinned and matches the 168-entry lock.

The new Hub namespace and immutable revision remain absent. Public bootstrap is therefore blocked. The SDK stages all publisher downloads into its isolated overlay and verifies exact hashes; it does not replace Kaggle's base Torch through ordinary SDK dependencies. Publisher availability and the complete public installation still require post-publication verification and manual Kaggle acceptance.
