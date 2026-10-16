# Additive Hugging Face reorganization

The current Hub commit is `8d7dd28f7c76c3b68e7326df81758411b7dc2680`, with 16 files. The T4 wheel remains at the repository root; the L4 native wheel is at `v0.3.0/l4x4/native/`. No wheelhouse is present. The target metadata and release index are staged in `release/hf-v032-staging/`.

1. Complete the third-party redistribution and notice review.
2. Use Hugging Face `CommitOperationCopy` for the two existing LFS wheels, from their immutable source revisions into `v0.2.0/native/` and `v0.3.2/l4x4/native/`. Keep the old paths.
3. Upload the 168 locally verified wheels from `../kaggle-vllm-v030-wheelhouse-cp312-cu129-v2/<group>/` into `v0.3.2/l4x4/wheelhouse/`; upload staged metadata and `releases.json`. No dummy files.
4. Read the resulting immutable commit tree and compare all 170 required L4 entries (168 dependencies, native, lock) with the lock. Download samples or the full set for independent checks as appropriate.
5. Pin the verified commit in the 0.3.2 SDK profile, rebuild SDK distributions, run CPU checks and then the user's manual Kaggle acceptance.

The exact 175 currently missing target paths are listed in `missing-public-l4-artifacts.json`. Existing paths are retained; deletion requires separate approval. Hugging Face supports server-side copying of LFS files, so the large native wheels need not be downloaded and re-uploaded solely to reorganize them.

Reference: [Hugging Face repository copy documentation](https://huggingface.co/docs/huggingface_hub/guides/repository).
