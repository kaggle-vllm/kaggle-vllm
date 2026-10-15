# Reuse the P31 Notebook Output for the native wheel

The preferred route avoids manually uploading either the 6.6 GB runtime-v3 tar or a separate 294 MB native wheel. Save/version the successful P31 final-v2 notebook **with its outputs**, then attach that saved Notebook Output to a **new** four-L4 competition notebook. The acceptance runner looks under `/kaggle/input/notebooks/` for, in priority order, the exact `vllm-0.30.0-cp312-cp312-linux_x86_64.whl` or the exact `p31-kaggle-vllm-v030-l4x4-final-v2.tar.gz`. If Kaggle mounts the output elsewhere, provide `--notebook-root`, `--native-wheel`, or `--p31-archive` explicitly.

The direct wheel must have SHA256 `9ce42cde3a8564d9670679dd1be11b0a8640fb88d22aef8f94299d5a6747a6e8`. For the tar route, the complete tar must have SHA256 `e82085d0c6f180d2f4f8eb8f52332a92051273715e0923082fc65fc8857ddad5`. The SDK reads only the exact regular member `p31-kaggle-vllm-v030-l4x4-final-v2/vllm-0.30.0-cp312-cp312-linux_x86_64.whl`, hashes it, and stages it into its owned runtime directory. It does not extract the rest of P31 or alter the native wheel.

Automatic selection rejects multiple candidates at the same priority. The runner never silently uses an older checkpoint tar or an upstream vLLM wheel. The corrected split-v2 wheel Datasets and Kaggle-provided Gemma Input Model are still required.

## Fallback if saved P31 output is unavailable

From this project checkout, prepare a dedicated **three-data-file** Kaggle Dataset locally:

```bash
python3.11 scripts/prepare_kaggle_native_wheel_dataset.py
```

The helper validates the canonical P31 tar and wheel before creating `dist/kaggle-vllm-v030-native-wheel-only/` with only the exact native wheel, `SHA256SUMS.txt`, and `SOURCE_IDENTITY.json`. It never uploads or reads credentials. The installed Kaggle CLI's `datasets create` command also needs a special `datasets-metadata.json` control file; the helper prints the later `kaggle datasets init` and `kaggle datasets create` sequence but executes neither. Review the control file and keep any eventual Dataset private unless redistribution permission is established. The three data files may instead be uploaded through Kaggle's Dataset UI later.

No native source is included in the small SDK package. No runtime-v3 upload is required for the split-v2 acceptance.
