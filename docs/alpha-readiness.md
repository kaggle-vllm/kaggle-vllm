# Alpha-readiness scorecard

This scorecard describes the public 0.2.0 SDK after local hardening, trusted
PyPI publication and the reviewed 2026-08-30 development-candidate Kaggle
evidence. The exact public package and focused existing-Qwen regression passed
their final dual-T4 gates on 2026-08-31.

| Area | Score | Current reason |
|---|---:|---|
| Technical implementation | 8.5 | focused SDK, dependency diagnostics and topology hardening |
| Architecture | 9.0 | explicit immutable delivery boundary and ownership model |
| Kaggle compatibility | 9.5 | exact-profile development candidate and public 0.2.0 package passed |
| Testing | 9.5 | broad CPU suite plus final public-package and Qwen dual-T4 evidence |
| CI | 9.0 | multi-version CPU, lint, static, links and packaging |
| Documentation | 9.0 | focused operator/developer/release set with evidence boundaries |
| Reproducibility | 8.5 | wheel reproduced byte-for-byte; sdist timestamp limitation documented exactly |
| Supply chain/provenance | 8.5 | immutable hash/revision and consolidated machine record |
| Security | 8.5 | explicit operations, path ownership and symlink checks |
| Performance evidence | 8.0 | controlled five-case T4 matrix executed; intentionally small sample/model |
| Packaging | 9.0 | small pure-Python SDK, no Torch/vLLM/CUDA dependency |
| Release engineering | 9.5 | trusted PyPI publication and final GPU evidence verified; immutable tag/release follows reviewed merge |
| Licensing/attribution | 9.0 | SDK/upstream/Qwen boundaries explicit |
| Public usability | 8.5 | quick start, doctor JSON, focused docs; GPU setup remains specialized |
| Maintainability | 8.5 | profile-oriented data and lightweight tooling |

The real-GPU gates are complete. The immutable tag and GitHub release follow
the reviewed evidence merge. The intentionally narrow hardware profile,
limited performance sample, and ongoing maintenance obligations are why the
scorecard is not inflated to 10/10.
