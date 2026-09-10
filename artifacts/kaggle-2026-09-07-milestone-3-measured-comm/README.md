# Milestone 3 canonical measured-communication evidence

Status: **CANONICAL_ACCEPTED**

This directory is the reviewed output of a fresh Kaggle T4 x2 execution of the
repository source notebook at commit
`4df0dd183d78e48739f509637934adf33b98ce82`. The downloaded executed notebook
is retained outside Git; it did not replace the repository source notebook.

## External evidence identities

- Kaggle evidence ZIP SHA256:
  `b5aedfdda06522d483042e55a4f9637dd582e72242936ee130cddeeae0ae5769`
- Executed p2 notebook SHA256:
  `a2bf24a37a63d12318c5188829310f95d8f3106733be5d301b5570bd63e18383`
- Repository source notebook SHA256:
  `28178ea0121448966f4b000df5343a279f3f5eaed82ad5e7a2e1f1bb203e9232`
- Native wheel SHA256:
  `5a9bd710b8a19fdd23abb3442baad892da977466f996334decd533a225f5fd0c`

The executed notebook's four substantive cell sources exactly match the
repository notebook. Kaggle appended one empty, unexecuted code cell. No
source-patch cell or Python error output was present.

## Measurement and result

Two Tesla T4 SM75 GPUs were connected as PHB peers, and `nvidia-smi topo -p2p`
reported read and write access as `OK` in both directions. The runtime was
CPython 3.12.13, PyTorch 2.10.0+cu128, CUDA toolkit 12.8.93, and NCCL 2.27.5.
NCCL INFO logging was enabled; algorithm and protocol were not forced.

The raw ledgers contain 10 payloads, five independent fresh-process
repetitions, and 100 timed collectives per payload/repetition: 5,000
critical-path observations. Each critical path equals the larger of the two
rank-local CUDA-event durations. The CSV and JSON ledgers are semantically
identical.

The payload-mean fit is:

`T_us(S) = 112.135079 us + S / 4.063916 GB/s`

The fitted intercept's 95% CI is `[93.111210, 131.158948] us`; the effective
beta 95% CI is `[4.051054, 4.076860] GB/s`. R² is `0.999984836`, RMSE over the
ten fitted payload means is `20.136460 us`, and the largest absolute residual
is `43.867687 us`. The intercept is a configuration-specific measured
all-reduce fit parameter, not classical transport alpha or universal PCIe/PHB
latency.

`M3_CANONICAL_REVIEW.json` records the independent integration audit. The
original `SHA256SUMS.txt` authenticates the 13 payload artifacts plus
`M3_PROVENANCE.json`; verify it from this directory with `sha256sum -c` and the
project `verify-hashes` command.

The earlier retry1 evidence is classified as `DEBUG_VALIDATION_RUN` and is not
part of this canonical directory.
