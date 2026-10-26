# Python 3.13 dual-T4 source synchronization plan

## Reconciliation base

- Local `main`, `origin/main`, and their merge base are the same commit: `44517c1ae81bff90a9913a1e35ed7224df0bffc2`.
- Local and remote `python313-t4-v020` branches did not exist at the start of this audit. The local Python 3.13 port is uncommitted on `main`.
- The annotated `v0.2.0` tag still points to commit `18e08cc69956491eda225297411946a99b41f328`. Neither the tag nor `main` will be moved.
- The external backup has 462 files; each matches its original SHA256 manifest. It contains byte-verified historical SDK artifacts and the cp312 CUDA wheel.

## File actions

| Group | Classification | Action |
| --- | --- | --- |
| Active SDK, tests, docs, T4 profile, CI | MODIFIED_FOR_PYTHON313 | Commit on new development branch after CPU checks and build. |
| Unchanged T4 release evidence and neutral metadata | IDENTICAL or HISTORICAL_IMMUTABLE | Preserve in Git unchanged. |
| New historical cp312 reference copies | HISTORICAL_IMMUTABLE | Commit as clearly labeled evidence; never use as active wheel configuration. |
| Prior local cleanup audit files | LOCAL_ONLY | Retain locally, omit from source branch because they contain machine-specific recovery paths and working-tree snapshots. |
| This synchronization inventory and plan | LOCAL_ONLY audit documentation | Commit the five inventories and plan after checking for sensitive data. Final reports are generated after publication and retained locally. |
| `dist/`, caches, temporary pip targets | INTENTIONALLY_DIFFERENT | Exclude from Git; retain only local build outputs and remove temporary install target after testing. |
| Historical `publish-pypi.yml` and cp312 build/overlay recipes in active locations | REMOTE_ONLY after transition | Delete only from development branch; preserve on `main`, tag, backup, and/or clearly labeled historical copies. |

## Remote artifact handling

- GitHub Release `v0.2.0`, its seven assets, and PyPI 0.2.0 files remain immutable. The backup bytes of the two historical SDK archives and native cp312 wheel match published SHA256 metadata.
- Hugging Face binaries at historical revision `f6b4f10de54924ed6fe9e28cceab84eca7276ab6` and current revision contain the cp312 wheel. Preserve without upload. Remote LFS SHA256 metadata and small checksum files agree with the byte-verified local backup; the large remote wheel bytes have not been separately downloaded.
- Hugging Face models contain the Qwen TP=2 rank-0/rank-1 sharded state, license, notice, and checksum manifest. Preserve without upload. Large remote checkpoint bytes have not been separately downloaded.
- The Python 3.13 cp313 native wheel, checksum, immutable Hub revision, dependency overlay, and GPU validation remain pending.

## Publication sequence

1. Run Python 3.13 CPU tests, static checks, clean SDK build, wheel metadata inspection, and isolated target import.
2. Create `python313-t4-v020` from current `origin/main` ancestry while retaining the verified local edits.
3. Stage only reviewed source, tests, documentation, historical T4 reference files, CI, and these inventories. Check staged diff for credentials, large binaries, and later GPU generation content.
4. Commit and normal-push the branch. Fetch and compare local/remote commit and tree identities.
5. Open an unmerged PR to `main` explaining the unvalidated native runtime and historical release preservation.
6. Re-read remote release, Hugging Face revisions, and PyPI metadata, then write local final synchronization reports.
