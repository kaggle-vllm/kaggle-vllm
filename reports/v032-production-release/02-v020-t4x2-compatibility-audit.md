# 0.2.0 dual T4 compatibility

The packaged historical profile remains `kaggle-t4x2-cu128`: CPython cp312, two Tesla T4/SM75 GPUs, TP=2, Torch 2.10.0+cu128, CUDA 12.8 and upstream vLLM v0.18.1 at `a26e8dc7ff2111a005144d775ecf9cebf56c45b2`.

The exact wheel `vllm-0.18.2.dev0+ga26e8dc7f.d20260822.cu128-cp312-cp312-linux_x86_64.whl` is 383922260 bytes and locally hashed to `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c` (PASS). The current Hub API reports the same LFS SHA256 at the historical root path. The profile still pins commit `f6b4f10de54924ed6fe9e28cceab84eca7276ab6`. Existing paths and checksum references were not edited.

The older public 0.2.0 package, its Qwen TP=2 sharded-state research and historical acceptance files remain separate from 0.3.2. The new manual notebook has only structural validation; a changed SDK cannot inherit its old GPU acceptance.

The exact PyPI 0.2.0 pure Python wheel was downloaded without installation, hashed against PyPI JSON (SHA256 `f3dce393c9e0bd43b9ba29a29ae14f9467857e5eea61390d41f512a52911fbbe`), and imported in a Python 3.11 subprocess. It reported version 0.2.0 and the T4 profile. This is a CPU packaging check only.
