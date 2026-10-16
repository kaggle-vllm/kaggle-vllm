# Kaggle network failure

`[Errno -3] Temporary failure in name resolution` is a DNS failure during pip networking. It does not establish any CUDA or wheel defect. The 0.3.2 `network-preflight` probes PyPI and Hugging Face separately with timeouts and reports DNS, TLS, proxy, host, and package-version failures. It also reports whether proxy variables and pip index keys are configured without exposing values.

In Kaggle Notebook settings, enable Internet, restart the session, and rerun the preflight before pip. Pip installation uses bounded retries and timeouts in the manual notebooks. A PyPI 404 before publication is expected. Dependency resolution failures require pip's own error log; disk exhaustion is checked separately by bootstrap. TLS verification and Kaggle network policy remain enabled.

## Publisher endpoints

The revised online runtime also needs `download.pytorch.org` and `flashinfer.ai` (and their HTTPS delivery redirects). The SDK network preflight now probes these hosts in addition to PyPI and Hugging Face. A Kaggle `[Errno -3] Temporary failure in name resolution` remains a network/DNS issue; users must enable Internet in Notebook settings. No TLS check or Kaggle restriction is bypassed.
