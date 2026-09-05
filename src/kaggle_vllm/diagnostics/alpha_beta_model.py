"""Communication-cost diagnostics model (Milestone 3).

Measured vs derived proxy vs optional hypothetical — never silent fake architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ModelArchitectureSpecs:
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    dtype_bytes_per_elem: int = 2

    @property
    def assumed_allreduce_calls_per_token(self) -> int:
        return 2 * self.num_layers


@dataclass(frozen=True)
class HypotheticalAlphaBetaAssumptions:
    """Caller-supplied only. No baked-in 7.91 us / 7.8 GB/s defaults."""

    alpha_us_per_collective: float
    beta_gb_s: float
    payload_mode: str  # "unknown" | "hypothetical_concurrency_as_batch"
    source_note: str

    def __post_init__(self) -> None:
        if self.alpha_us_per_collective < 0 or self.beta_gb_s <= 0:
            raise ValueError("hypothetical alpha must be >= 0 and beta must be > 0")
        if self.payload_mode not in {"unknown", "hypothetical_concurrency_as_batch"}:
            raise ValueError(f"unsupported payload_mode={self.payload_mode!r}")
        if not self.source_note.strip():
            raise ValueError(
                "hypothetical source_note is required (no unsourced defaults)"
            )


@dataclass(frozen=True)
class CellCommDiagnosis:
    label: str
    model_key: str
    concurrency: int

    measured_tp1_tok_s: float
    measured_tp2_tok_s: float
    measured_delta_percent: float
    measured_tp1_ms_per_tok: float | None
    measured_tp2_ms_per_tok: float | None
    # signed: TP2 - TP1 in ms/tok service time (negative => TP2 faster)
    measured_service_time_delta_ms_per_tok: float | None
    # positive-only excess when TP2 slower; None otherwise
    measured_excess_ms_per_tok: float | None

    assumed_num_layers: int
    assumed_collectives_per_token: int

    observed_excess_us_per_collective_proxy: float | None
    proxy_caveat: str
    measured_regime: str  # TP2_SLOWER | TP2_FASTER | TP_EQUIVALENT | INVALID

    hypothetical_enabled: bool
    hypothetical_t_comm_ms_per_tok: float | None
    hypothetical_residual_ms_per_tok: float | None
    hypothetical_note: str
    summary: str


class UnsupportedModelArchitectureError(ValueError):
    """Unknown model_key without explicit architecture."""


class AlphaBetaCommModel:
    SPECS: ClassVar[dict[str, ModelArchitectureSpecs]] = {
        "opt-125m": ModelArchitectureSpecs("opt-125m", 12, 768, 12),
        "qwen2.5-3b": ModelArchitectureSpecs("qwen2.5-3b", 36, 2048, 16),
    }

    # relative tok/s equality tolerance
    THROUGHPUT_EQ_RTOL: ClassVar[float] = 1e-6

    def __init__(
        self,
        hypothetical: HypotheticalAlphaBetaAssumptions | None = None,
        enable_hypothetical: bool = False,
        architecture_overrides: dict[str, ModelArchitectureSpecs] | None = None,
    ) -> None:
        if enable_hypothetical and hypothetical is None:
            raise ValueError(
                "enable_hypothetical=True requires explicit HypotheticalAlphaBetaAssumptions "
                "(no default alpha/beta constants)"
            )
        self.hypothetical = hypothetical
        self.enable_hypothetical = enable_hypothetical
        self._overrides = architecture_overrides or {}

    def get_specs(self, model_key: str) -> ModelArchitectureSpecs:
        key = model_key.lower().strip()
        if key in self._overrides:
            return self._overrides[key]
        if key in self.SPECS:
            return self.SPECS[key]
        known = ", ".join(sorted(set(self.SPECS) | set(self._overrides)))
        raise UnsupportedModelArchitectureError(
            f"unsupported model_key={model_key!r}. known={known}. "
            "Pass architecture_overrides=... instead of silent fallback."
        )

    @staticmethod
    def ms_per_tok_from_throughput(tok_s: float) -> float | None:
        if tok_s <= 0:
            return None
        return 1000.0 / tok_s

    def _regime(self, tp1: float, tp2: float) -> str:
        if tp1 <= 0 or tp2 <= 0:
            return "INVALID"
        # equality on throughput space
        scale = max(abs(tp1), abs(tp2), 1.0)
        if abs(tp1 - tp2) <= self.THROUGHPUT_EQ_RTOL * scale:
            return "TP_EQUIVALENT"
        return "TP2_FASTER" if tp2 > tp1 else "TP2_SLOWER"

    def excess_service_time_proxy(
        self,
        tp1_tok_s: float,
        tp2_tok_s: float,
        collectives_per_token: int,
    ) -> tuple[
        float | None, float | None, float | None, float | None, float | None, str
    ]:
        t1 = self.ms_per_tok_from_throughput(tp1_tok_s)
        t2 = self.ms_per_tok_from_throughput(tp2_tok_s)
        if t1 is None or t2 is None or collectives_per_token <= 0:
            return t1, t2, None, None, None, "invalid throughput or collective count"

        signed_delta = t2 - t1
        if signed_delta <= 0:
            return (
                t1,
                t2,
                signed_delta,
                None,
                None,
                (
                    "No positive excess service time (TP2 not slower than TP1 in ms/tok). "
                    "Collective excess-time proxy undefined."
                ),
            )

        us_per = (signed_delta / collectives_per_token) * 1000.0
        caveat = (
            "DERIVED PROXY only: positive excess_ms_per_tok / assumed_collectives. "
            "NOT isolated transport alpha. Mixes compute, scheduling, graph/eager, memory, beta*S."
        )
        return t1, t2, signed_delta, signed_delta, us_per, caveat

    def hypothetical_t_comm_ms(
        self,
        specs: ModelArchitectureSpecs,
        tp_size: int,
        concurrency: int,
    ) -> tuple[float | None, str]:
        if not self.enable_hypothetical or self.hypothetical is None:
            return None, "Hypothetical path disabled or not configured."

        if tp_size <= 1:
            return 0.0, "tp_size<=1 => hypothetical T_comm = 0."

        hyp = self.hypothetical
        n = specs.assumed_allreduce_calls_per_token
        alpha_ms = hyp.alpha_us_per_collective / 1000.0
        ring = 2.0 * (tp_size - 1) / tp_size

        if hyp.payload_mode == "unknown":
            t = n * alpha_ms
            note = (
                f"HYPOTHETICAL latency-only N*alpha with caller alpha="
                f"{hyp.alpha_us_per_collective} us. Payload omitted. Source: {hyp.source_note}"
            )
            return t, note

        # hypothetical_concurrency_as_batch
        s_bytes = concurrency * specs.hidden_size * specs.dtype_bytes_per_elem
        beta_bytes_per_ms = hyp.beta_gb_s * 1e6
        term_b = (ring * s_bytes) / beta_bytes_per_ms
        t = n * (alpha_ms + term_b)
        note = (
            "HYPOTHETICAL WEAK: S uses request concurrency as decode batch "
            f"(not validated for online vLLM). Source: {hyp.source_note}"
        )
        return t, note

    def diagnose_cell(
        self,
        *,
        label: str,
        model_key: str,
        tp1_tok_s: float,
        tp2_tok_s: float,
        concurrency: int = 1,
        comparison_is_tp_controlled: bool = True,
        architecture: ModelArchitectureSpecs | None = None,
    ) -> CellCommDiagnosis:
        specs = architecture if architecture is not None else self.get_specs(model_key)
        n_coll = specs.assumed_allreduce_calls_per_token

        if not comparison_is_tp_controlled:
            delta = (
                ((tp2_tok_s - tp1_tok_s) / tp1_tok_s) * 100.0 if tp1_tok_s > 0 else 0.0
            )
            return CellCommDiagnosis(
                label=label,
                model_key=model_key,
                concurrency=concurrency,
                measured_tp1_tok_s=tp1_tok_s,
                measured_tp2_tok_s=tp2_tok_s,
                measured_delta_percent=delta,
                measured_tp1_ms_per_tok=self.ms_per_tok_from_throughput(tp1_tok_s),
                measured_tp2_ms_per_tok=self.ms_per_tok_from_throughput(tp2_tok_s),
                measured_service_time_delta_ms_per_tok=None,
                measured_excess_ms_per_tok=None,
                assumed_num_layers=specs.num_layers,
                assumed_collectives_per_token=n_coll,
                observed_excess_us_per_collective_proxy=None,
                proxy_caveat="N/A — not a controlled TP1 vs TP2 experiment.",
                measured_regime="INVALID",
                hypothetical_enabled=False,
                hypothetical_t_comm_ms_per_tok=None,
                hypothetical_residual_ms_per_tok=None,
                hypothetical_note="Skipped.",
                summary=(
                    f"MEASURED only for {label}: not a controlled TP1/TP2 comparison. "
                    "Do not derive collective excess-time proxy or PHB transport claims."
                ),
            )

        if tp1_tok_s <= 0 or tp2_tok_s <= 0:
            return CellCommDiagnosis(
                label=label,
                model_key=model_key,
                concurrency=concurrency,
                measured_tp1_tok_s=tp1_tok_s,
                measured_tp2_tok_s=tp2_tok_s,
                measured_delta_percent=0.0,
                measured_tp1_ms_per_tok=None,
                measured_tp2_ms_per_tok=None,
                measured_service_time_delta_ms_per_tok=None,
                measured_excess_ms_per_tok=None,
                assumed_num_layers=specs.num_layers,
                assumed_collectives_per_token=n_coll,
                observed_excess_us_per_collective_proxy=None,
                proxy_caveat="Invalid throughput.",
                measured_regime="INVALID",
                hypothetical_enabled=self.enable_hypothetical,
                hypothetical_t_comm_ms_per_tok=None,
                hypothetical_residual_ms_per_tok=None,
                hypothetical_note="Skipped.",
                summary="Invalid or missing throughput measurements.",
            )

        delta_pct = ((tp2_tok_s - tp1_tok_s) / tp1_tok_s) * 100.0
        regime = self._regime(tp1_tok_s, tp2_tok_s)
        t1, t2, signed_delta, excess_ms, us_proxy, caveat = (
            self.excess_service_time_proxy(tp1_tok_s, tp2_tok_s, n_coll)
        )

        hyp_t, hyp_note = self.hypothetical_t_comm_ms(
            specs, tp_size=2, concurrency=concurrency
        )
        residual = None
        if hyp_t is not None and excess_ms is not None:
            residual = excess_ms - hyp_t

        summary = (
            f"MEASURED: TP1={tp1_tok_s:.4f} tok/s, TP2={tp2_tok_s:.4f} tok/s "
            f"({delta_pct:+.2f}%), regime={regime}. "
            f"Assumed architecture: {specs.num_layers} layers, "
            f"{n_coll} collectives/token (2 per layer). "
        )
        if us_proxy is not None:
            summary += (
                f"DERIVED PROXY positive excess ~{us_proxy:.2f} us/collective "
                f"(not isolated alpha). "
            )
        summary += (
            "PHB may be observed on host topology evidence; M1/M2 do not isolate "
            "NCCL/PCIe as sole cause of TP deltas."
        )

        return CellCommDiagnosis(
            label=label,
            model_key=model_key,
            concurrency=concurrency,
            measured_tp1_tok_s=tp1_tok_s,
            measured_tp2_tok_s=tp2_tok_s,
            measured_delta_percent=delta_pct,
            measured_tp1_ms_per_tok=t1,
            measured_tp2_ms_per_tok=t2,
            measured_service_time_delta_ms_per_tok=signed_delta,
            measured_excess_ms_per_tok=excess_ms,
            assumed_num_layers=specs.num_layers,
            assumed_collectives_per_token=n_coll,
            observed_excess_us_per_collective_proxy=us_proxy,
            proxy_caveat=caveat,
            measured_regime=regime,
            hypothetical_enabled=self.enable_hypothetical,
            hypothetical_t_comm_ms_per_tok=hyp_t,
            hypothetical_residual_ms_per_tok=residual,
            hypothetical_note=hyp_note,
            summary=summary,
        )
