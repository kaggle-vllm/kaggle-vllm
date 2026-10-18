# Hugging Face repository audit

The official Hub API reported public **model** repository `waqasm86/kaggle-vllm-binaries`, `main` revision `8d7dd28f7c76c3b68e7326df81758411b7dc2680`, 16 files, and LFS/Xet-backed native artifact entries. Historical T4 files and root SDK 0.1.x files are present. The L4 native wheel exists at `v0.3.0/l4x4/native/vllm-0.30.0-cp312-cp312-linux_x86_64.whl` with recorded size/digest matching the local native wheel. No v0.3.2 namespace or 168-wheel dependency closure appeared in the tree. Authenticated account identity was `waqasm86`; write capability was not exercised.

The repository card says native compiled-component review is incomplete and the dependency closure is not approved for public mirroring. Local `release/v032-redistribution-clearance.json` records overall `BLOCKED`, 0 of 168 third-party wheels reviewed and native compiled components `BLOCKED`. Publicly copying the native wheel and dependency wheels now would bypass this explicit gate. The existing historical objects remain untouched.

A Hub `resolve/<commit>/<file>` URL serves a known file, but this model repository has no verified PEP 503 `/<normalized-project>/` index pages. A separate hosted static index might be possible but has not been created or validated. There is no approved index URL to configure in pip. No direct immutable v0.3.2 SDK URL exists until publication.
