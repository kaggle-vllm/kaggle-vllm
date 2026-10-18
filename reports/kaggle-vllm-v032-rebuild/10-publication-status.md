# Publication status

`LOCAL_CANDIDATE_BUILT; PENDING_KAGGLE_ACCEPTANCE; PUBLICATION_BLOCKED`.

- Hugging Face repository main observed at `8d7dd28f7c76c3b68e7326df81758411b7dc2680`; no v0.3.2 SDK or closure uploaded and no release revision pinned.
- GitHub: no v0.3.2 draft release or assets created. No tag moved, GitHub settings changed, or remote source pushed.
- Blocking gate: `release/v032-redistribution-clearance.json` has `overall=BLOCKED`, 0/168 third-party dependency wheels reviewed, native compiled components blocked. Hub card independently warns that native review and closure mirroring are incomplete.
- Technical gap: approved Hub repository lacks all 168 dependency wheels and SDK 0.3.2 files, so HF-only runtime publication cannot be represented as complete.
- Needed from owner: documented redistribution clearance for the exact native wheel and third-party closure, a reviewed immutable Hub artifact layout, and approval to proceed with the resulting public mirroring. Then verify actual Hub commits/checksums, pin that revision, rebuild distributions, create a GitHub draft, upload assets, and leave final Kaggle acceptance to the owner.
