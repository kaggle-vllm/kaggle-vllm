# Local repository audit

- Working release branch: `release/v0.3.2-l4x4-public-runtime`; starting commit: `cfa5659e479dfdda8a9ac0ac05d88562ade4cbf5` (2026-10-16).
- The original `kaggle-vllm` main checkout has substantial uncommitted 0.3.0 work and is behind origin/main by 15 commits. It was left untouched. The source branch `fix/v031-offline-base-deps` had an unrelated edit to `competition/paper-track/writeup.md`; it was also left untouched.
- This isolated worktree is based on the newest inspected 0.3.1 line, `fix/v031-offline-base-deps`, which includes the accepted SDK repair, competition gates, GPU evidence and offline-base-deps work.
- The repository includes src, tests, scripts, docs, release, artifacts, research, Kaggle notebooks and GitHub workflows. `dist` in this worktree contains only local 0.3.2 SDK candidates. Historical tags were not changed.
- Python 3.11.15 and pip 26.2.1 were used. No virtual environment, Conda, uv, Ruff, sudo or CUDA toolkit change was used.
