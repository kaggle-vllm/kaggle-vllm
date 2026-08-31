# Alpha-readiness scorecard

This scorecard describes the public 0.2.0 SDK after local hardening, trusted
PyPI publication and the reviewed 2026-08-30 development-candidate Kaggle
evidence. Final published-package GPU acceptance remains pending.

| Area | Score | Current reason |
|---|---:|---|
| Technical implementation | 8.5 | focused SDK, dependency diagnostics and topology hardening |
| Architecture | 9.0 | explicit immutable delivery boundary and ownership model |
| Kaggle compatibility | 9.0 | exact-profile 0.2 development candidate passed; published-package rerun pending |
| Testing | 9.0 | broad CPU suite plus fresh dual-T4 acceptance; final package remains separate |
| CI | 9.0 | multi-version CPU, lint, static, links and packaging |
| Documentation | 9.0 | focused operator/developer/release set with evidence boundaries |
| Reproducibility | 8.5 | wheel reproduced byte-for-byte; sdist timestamp limitation documented exactly |
| Supply chain/provenance | 8.5 | immutable hash/revision and consolidated machine record |
| Security | 8.5 | explicit operations, path ownership and symlink checks |
| Performance evidence | 8.0 | controlled five-case T4 matrix executed; intentionally small sample/model |
| Packaging | 9.0 | small pure-Python SDK, no Torch/vLLM/CUDA dependency |
| Release engineering | 9.0 | trusted PyPI publication verified; GPU acceptance/tag/release remain gated |
| Licensing/attribution | 9.0 | SDK/upstream/Qwen boundaries explicit |
| Public usability | 8.5 | quick start, doctor JSON, focused docs; GPU setup remains specialized |
| Maintainability | 8.5 | profile-oriented data and lightweight tooling |

Remaining release gates are a fresh dual-T4 acceptance run against the public
`kaggle-vllm==0.2.0` package and a focused existing-Qwen TP=2 sharded-state
regression if it is required for final scope. The immutable tag and GitHub
release follow those gates. These remaining operations are why the scorecard
is not inflated to 10/10.
