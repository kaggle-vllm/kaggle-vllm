"""CPU-only research evidence analysis for the immutable 0.2.0 baseline."""

from .comparison import build_model_vs_observed
from .measured_comm import (
    REQUIRED_PAYLOAD_BYTES,
    fit_communication_model,
    load_allreduce_csv,
    load_allreduce_json,
    summarize_allreduce,
)
from .provenance import sha256_file, verify_sha256_manifest

__all__ = [
    "REQUIRED_PAYLOAD_BYTES",
    "build_model_vs_observed",
    "fit_communication_model",
    "load_allreduce_csv",
    "load_allreduce_json",
    "sha256_file",
    "summarize_allreduce",
    "verify_sha256_manifest",
]
