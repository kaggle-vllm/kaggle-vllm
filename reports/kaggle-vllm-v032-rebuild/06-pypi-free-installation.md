# PyPI-free installation

## Option A: configured simple index

Unavailable. The current Hugging Face model repository exposes file `resolve` URLs but no verified PEP 503 simple index pages. Do not configure pip to an invented endpoint, and do not run unconfigured `pip install kaggle-vllm` for this candidate.

## Option B: exact local offline Inputs

Mount the locally built SDK wheel, `RELEASE_MANIFEST.json`, native wheel and four v2 ZIPs as Kaggle Inputs. Put the four ZIPs beside the native wheel, as in the local `kaggle-upload` directory. The manual notebook embeds the installer script, so no separate helper mount is required. Use the actual mounted paths in these commands:

```bash
/usr/bin/python3.12 -m pip install --no-index --no-deps /kaggle/input/SDK_DATASET/kaggle_vllm-0.3.2-py3-none-any.whl
/usr/bin/python3.12 /kaggle/working/install_global_v032_offline.py --input-root /kaggle/input/RUNTIME_DATASET --sdk-wheel /kaggle/input/SDK_DATASET/kaggle_vllm-0.3.2-py3-none-any.whl --release-manifest /kaggle/input/SDK_DATASET/RELEASE_MANIFEST.json
/usr/bin/python3.12 /kaggle/working/install_global_v032_offline.py --input-root /kaggle/input/RUNTIME_DATASET --sdk-wheel /kaggle/input/SDK_DATASET/kaggle_vllm-0.3.2-py3-none-any.whl --release-manifest /kaggle/input/SDK_DATASET/RELEASE_MANIFEST.json --apply --replace-global-cuda-stack
```

The first installer invocation verifies and stages but does not run pip. The second explicitly replaces Kaggle's preinstalled Torch/cu128 stack with the pinned cu129 closure. The script itself invokes pip with `--no-index --no-deps --only-binary=:all: --require-hashes`; no source build or fallback index is allowed. This installation has not been executed on Kaggle.

## Direct immutable Hub URL

Unavailable because SDK 0.3.2 is not published. After publication, use `/usr/bin/python3.12 -m pip install --no-index --no-deps "<exact resolve URL with published commit>"`; publish and verify the full dependency closure separately before describing this as a complete runtime install. No placeholder is presented as a working URL.
