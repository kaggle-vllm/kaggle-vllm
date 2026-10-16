# Hugging Face publication readiness

The current Hub tree has 16 files at `8d7dd28f7c76c3b68e7326df81758411b7dc2680`. Both exact native wheels already exist under historic paths. The target 0.3.2 namespace lacks all 168 dependencies, native copy and metadata. Local metadata is staged in `release/hf-v032-staging/`; all wheel bytes are locally verified, and no large binary was copied into Git.

Publication is BLOCKED by the third-party redistribution review and the user's required authorization. After approval, perform additive copies/uploads, verify the immutable resulting commit and only then pin the SDK. Do not delete root T4 or v0.3.0 L4 paths.
