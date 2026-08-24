# Validation plan

The original staged-gate plan has been completed on Kaggle. The resulting
evidence and outcomes are documented in [validation.md](validation.md).

Future compatibility runs should repeat, in order:

1. artifact checksum and binary inspection;
2. staged imports without dependency resolution;
3. dependency overlay validation while preserving system Torch;
4. raw NCCL all-reduce;
5. single-GPU inference;
6. TP=2 inference;
7. real target-model inference;
8. persistent `sharded_state` save and fresh-engine reload;
9. OpenAI-compatible endpoint checks;
10. only then performance experimentation.

Record full runtime identities and never convert a skipped or local CPU test
into a claimed GPU pass.
