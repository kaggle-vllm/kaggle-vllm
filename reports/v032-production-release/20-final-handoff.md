# 0.3.2 release candidate handoff

The 0.3.2 source and pure Python SDK distributions are prepared in this worktree. The exact 168-wheel local inference closure and both native wheels passed SHA256 checks. The Hub inventory has 16 files and lacks the public 0.3.2 wheelhouse. The 0.3.2 profile deliberately has no immutable revision; public bootstrap fails clearly. Both manual notebooks are output-free and were not executed.

[Draft PR #39](https://github.com/kaggle-vllm/kaggle-vllm/pull/39) contains the source and release plan for review. It is not a published 0.3.2 release.

**Production blockers:** third-party binary redistribution review, approved/additive Hugging Face publication of 168 wheels and metadata, immutable Hub pin, final SDK rebuild, source PR, PyPI/GitHub publication authorization, and fresh manual Kaggle acceptance. No 0.3.2 GPU or Gemma acceptance is claimed.

Historical T4 0.2.0 wheel SHA256: `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`. Preserved L4 native wheel SHA256: `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`. Local SDK candidate checksums: `release/v032-SHA256SUMS.txt`.
