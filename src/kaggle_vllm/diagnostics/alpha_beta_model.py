"""Analytical helpers for TP communication-cost *diagnostics* (Milestone 3).

IMPORTANT — scientific boundaries
---------------------------------
* Throughput TP1 vs TP2 differences are MEASURED.
* Dividing excess service time by an assumed collective count yields a
  DERIVED PROXY only. It is NOT an isolated alpha (α) from the classical
  α–β model, because the residual still mixes compute, scheduling,
  graph/eager effects, memory traffic, and any bandwidth term (β·S).
* Optional α–β evaluation is HYPOTHETICAL unless backed by direct
  collective timers or scheduler batch traces (not present in M1/M2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ModelArchitectureSpecs:
    """Transformer shape used only to count *assumed* Megatron-style collectives."""

    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    dtype_bytes_per_elem: int = 2  # FP16/BF16

    @property
    def assumed_allreduce_calls_per_token(self) -> int:
        """Megatron TP: typically 2 AllReduce calls per layer (attn out + MLP down)."""
        return 2 * self.num_layers


@dataclass(frozen=True)
class HypotheticalAlphaBetaAssumptions:
    """Optional what-if parameters. NOT calibrated from repo evidence unless stated."""

    alpha_us_per_collective: float
    beta_gb_s: float
    payload_mode: str  # "unknown" | "hypothetical_concurrency_as_batch"
    source_note: str


DEFAULT_HYPOTHETICAL = HypotheticalAlphaBetaAssumptions(
    alpha_us_per_collective=7.91,
    beta_gb_s=7.8,
    payload_mode="unknown",
    source_note=(
        "Defaults are literature/PCIe-order-of-magnitude assumptions only. "
        "They are NOT fitted transport parameters from M1/M2 timers. "
        "beta~7.8 GB/s approximates a PCIe Gen3 x8 unidirectional peak class link; "
        "not measured on the Kaggle host in-repo."
    ),
)


@dataclass(frozen=True)
class CellCommDiagnosis:
    """One concurrency (or offline) cell: measured facts + optional proxy/hypothesis."""

    # identity
    label: str
    model_key: str
    concurrency: int

    # MEASURED
    measured_tp1_tok_s: float
    measured_tp2_tok_s: float
    measured_delta_percent: float
    measured_tp1_ms_per_tok: float | None
    measured_tp2_ms_per_tok: float | None
    measured_excess_ms_per_tok: float | None  # max(0, tp2 - tp1) in ms/tok service time

    # architecture assumptions (for collective count only)
    assumed_num_layers: int
    assumed_collectives_per_token: int

    # DERIVED PROXY (not alpha)
    observed_excess_us_per_collective_proxy: float | None
    proxy_caveat: str

    # regime label from measurements only
    measured_regime: str  # "TP2_SLOWER" | "TP2_FASTER" | "INVALID"

    # optional hypothetical α–β (explicitly non-causal unless validated)
    hypothetical_enabled: bool
    hypothetical_t_comm_ms_per_tok: float | None
    hypothetical_residual_ms_per_tok: float | None
    hypothetical_note: str

    # prose — observational
    summary: str


class AlphaBetaCommModel:
    """Diagnostics model with strict measured vs proxy vs hypothetical layers."""

    SPECS: ClassVar[dict[str, ModelArchitectureSpecs]] = {
        "opt-125m": ModelArchitectureSpecs(
            name="opt-125m",
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
        ),
        "qwen2.5-3b": ModelArchitectureSpecs(
            name="qwen2.5-3b",
            num_layers=36,
            hidden_size=2048,
            num_attention_heads=16,
        ),
    }

    def __init__(
        self,
        hypothetical: HypotheticalAlphaBetaAssumptions | None = None,
        enable_hypothetical: bool = False,
    ) -> None:
        self.hypothetical = hypothetical or DEFAULT_HYPOTHETICAL
        self.enable_hypothetical = enable_hypothetical

    def get_specs(self, model_key: str) -> ModelArchitectureSpecs:
        key = model_key.lower()
        if key not in self.SPECS:
            return ModelArchitectureSpecs(
                name=key,
                num_layers=24,
                hidden_size=2048,
                num_attention_heads=16,
            )
        return self.SPECS[key]

    @staticmethod
    def ms_per_tok_from_throughput(tok_s: float) -> float | None:
        if tok_s <= 0:
            return None
        return 1000.0 / tok_s

    def excess_service_time_proxy(
        self,
        tp1_tok_s: float,
        tp2_tok_s: float,
        collectives_per_token: int,
    ) -> tuple[float | None, float | None, float | None, str]:
        """Return (tp1_ms, tp2_ms, excess_ms, us_per_collective_proxy)."""
        t1 = self.ms_per_tok_from_throughput(tp1_tok_s)
        t2 = self.ms_per_tok_from_throughput(tp2_tok_s)
        if t1 is None or t2 is None or collectives_per_token <= 0:
            return t1, t2, None, None, "invalid throughput or collective count"

        excess_ms = t2 - t1
        if excess_ms <= 0:
            return (
                t1,
                t2,
                excess_ms,
                None,
                (
                    "TP2 service time is not higher than TP1; "
                    "no positive excess-time proxy to attribute to collectives."
                ),
            )

        us_per = (excess_ms / collectives_per_token) * 1000.0
        caveat = (
            "DERIVED PROXY only: excess_ms_per_tok / assumed_collectives. "
            "This is NOT isolated alpha (α). Residual still includes compute, "
            "scheduling, graph/eager, memory, and β·S terms."
        )
        return t1, t2, excess_ms, us_per, caveat

    def hypothetical_t_comm_ms(
        self,
        specs: ModelArchitectureSpecs,
        tp_size: int,
        concurrency: int,
    ) -> tuple[float | None, str]:
        if not self.enable_hypothetical:
            return None, "Hypothetical α–β path disabled (default)."

        if tp_size <= 1:
            return 0.0, "tp_size<=1 => T_comm hypothesis = 0."

        hyp = self.hypothetical
        n = specs.assumed_allreduce_calls_per_token
        alpha_ms = hyp.alpha_us_per_collective / 1000.0
        ring = 2.0 * (tp_size - 1) / tp_size

        if hyp.payload_mode == "unknown":
            # latency-only hypothesis: N * alpha  (still NOT measured alpha)
            t = n * alpha_ms
            note = (
                f"HYPOTHETICAL latency-only: N_collectives*alpha_assumed "
                f"({n}*{hyp.alpha_us_per_collective} us). "
                f"Payload/β term omitted (payload_mode=unknown). {hyp.source_note}"
            )
            return t, note

        if hyp.payload_mode == "hypothetical_concurrency_as_batch":
            # EXPLICIT unsupported equality for serving — labeled assumption
            s_bytes = concurrency * specs.hidden_size * specs.dtype_bytes_per_elem
            beta_bytes_per_ms = hyp.beta_gb_s * 1e6
            term_b = (
                (ring * s_bytes) / beta_bytes_per_ms if beta_bytes_per_ms > 0 else 0.0
            )
            t = n * (alpha_ms + term_b)
            note = (
                "HYPOTHETICAL and WEAK: sets S using request concurrency as if it were "
                "instantaneous decode batch size. Online vLLM continuous batching does NOT "
                f"guarantee that equality. {hyp.source_note}"
            )
            return t, note

        return None, f"Unknown payload_mode={hyp.payload_mode!r}."

    def diagnose_cell(
        self,
        *,
        label: str,
        model_key: str,
        tp1_tok_s: float,
        tp2_tok_s: float,
        concurrency: int = 1,
        comparison_is_tp_controlled: bool = True,
    ) -> CellCommDiagnosis:
        specs = self.get_specs(model_key)
        n_coll = specs.assumed_allreduce_calls_per_token

        if not comparison_is_tp_controlled:
            return CellCommDiagnosis(
                label=label,
                model_key=model_key,
                concurrency=concurrency,
                measured_tp1_tok_s=tp1_tok_s,
                measured_tp2_tok_s=tp2_tok_s,
                measured_delta_percent=(
                    ((tp2_tok_s - tp1_tok_s) / tp1_tok_s) * 100.0
                    if tp1_tok_s > 0
                    else 0.0
                ),
                measured_tp1_ms_per_tok=self.ms_per_tok_from_throughput(tp1_tok_s),
                measured_tp2_ms_per_tok=self.ms_per_tok_from_throughput(tp2_tok_s),
                measured_excess_ms_per_tok=None,
                assumed_num_layers=specs.num_layers,
                assumed_collectives_per_token=n_coll,
                observed_excess_us_per_collective_proxy=None,
                proxy_caveat="N/A — comparison is not a controlled TP1 vs TP2 experiment.",
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
        t1, t2, excess_ms, us_proxy, caveat = self.excess_service_time_proxy(
            tp1_tok_s, tp2_tok_s, n_coll
        )

        if delta_pct < 0:
            regime = "TP2_SLOWER"
        else:
            regime = "TP2_FASTER"

        hyp_t, hyp_note = self.hypothetical_t_comm_ms(
            specs, tp_size=2, concurrency=concurrency
        )
        residual = None
        if hyp_t is not None and excess_ms is not None:
            # residual = measured excess - hypothetical T_comm (sign-aware)
            residual = excess_ms - hyp_t

        summary = (
            f"MEASURED: TP1={tp1_tok_s:.4f} tok/s, TP2={tp2_tok_s:.4f} tok/s "
            f"({delta_pct:+.2f}%). "
            f"Assumed architecture: {specs.num_layers} layers, "
            f"{n_coll} collectives/token (2 per layer). "
        )
        if us_proxy is not None:
            summary += (
                f"DERIVED PROXY excess service time ~{us_proxy:.2f} us/collective "
                f"(not isolated alpha). "
            )
        summary += (
            "Topology PHB was observed on the host, but M1/M2 do not isolate NCCL/PCIe "
            "as the sole cause of TP deltas."
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
