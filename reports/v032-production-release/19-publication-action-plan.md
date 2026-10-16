# Publication action plan

1. Resolve legal/provenance decisions for the native compiled components and all 168 mirrored wheels; finalize notices.
2. Obtain user authorization for additive Hugging Face publication. Copy the two existing LFS native wheels server-side, upload the verified 168 wheels and staged metadata/index, and preserve old paths.
3. Re-inventory the immutable Hub commit. Fail if any filename, size or SHA256 differs. Pin that commit in the SDK profile and update release metadata.
4. Rebuild 0.3.2 with Python 3.11, run CPU tests, twine and reproducibility checks, and open a source PR for review.
5. Obtain user authorization for PyPI Trusted Publishing and GitHub v0.3.2 release publication. Upload exact reviewed SDK files, checksums and notes. Never upload a credential.
6. The user manually runs both Kaggle notebooks. Review the resulting evidence. Mark 0.3.2 accepted only if all mandatory gates pass.

No remote mutation is performed by this report. The user must explicitly authorize the HF upload, PyPI upload and GitHub release publication as separate visible actions.
