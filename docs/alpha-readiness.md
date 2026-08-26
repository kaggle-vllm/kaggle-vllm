# Alpha-readiness scorecard

This scorecard describes the 0.2.0 development branch after local hardening.
Scores are evidence-based and should be updated after CI and Kaggle acceptance.

| Area | Score | Current reason |
|---|---:|---|
| Technical implementation | 8.5 | focused SDK, dependency diagnostics and topology hardening |
| Architecture | 9.0 | explicit immutable delivery boundary and ownership model |
| Kaggle compatibility | 8.5 | strong exact-profile history; new code awaiting acceptance |
| Testing | 8.5 | broad CPU unit suite; real GPU tests necessarily separate |
| CI | 9.0 | multi-version CPU, lint, static, links and packaging |
| Documentation | 9.0 | focused operator/developer/release set with evidence boundaries |
| Reproducibility | 8.0 | source/build identities captured; native rebuild not repeated |
| Supply chain/provenance | 8.5 | immutable hash/revision and consolidated machine record |
| Security | 8.5 | explicit operations, path ownership and symlink checks |
| Performance evidence | 5.0 | serious harness exists; no new T4 measurements yet |
| Packaging | 9.0 | small pure-Python SDK, no Torch/vLLM/CUDA dependency |
| Release engineering | 8.5 | staged checklist/workflow; 0.2.0 not published |
| Licensing/attribution | 9.0 | SDK/upstream/Qwen boundaries explicit |
| Public usability | 8.5 | quick start, doctor JSON, focused docs; GPU setup remains specialized |
| Maintainability | 8.5 | profile-oriented data and lightweight tooling |

The principal blocker to a true 10/10 alpha is real execution of the new
dependency doctor, acceptance notebook and TP1/TP2 benchmark matrix on a fresh
Kaggle dual-T4 runtime, followed by review of those results. Repeated clean SDK
builds are also needed before claiming byte-for-byte reproducibility.
