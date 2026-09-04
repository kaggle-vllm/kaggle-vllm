"""
Alpha-Beta Communication Cost Model for Tensor Parallelism (Megatron-style AllReduce).

Distinguishes:
- MEASURED QUANTITIES: Execution time, throughput, batch size, token count from JSON artifacts.
- INFERRED QUANTITIES: Per-step latency overhead (alpha), bandwidth capacity (beta),
  and estimated AllReduce bus contention on PCIe Host Bridge (PHB).
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass(frozen=True)
class ModelArchitectureSpecs:
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    dtype_bytes_per_elem: int = 2  # FP16 / BF16 default

    @property
    def allreduce_calls_per_token(self) -> int:
        # Megatron TP executes 2 AllReduce calls per Transformer layer:
        # 1. After Self-Attention Projection (o_proj)
        # 2. After MLP Down-Projection (down_proj)
        return 2 * self.num_layers

    def payload_bytes_per_allreduce(self, batch_size: int = 1) -> int:
        # Message size S = Batch_Size * Hidden_Size * Bytes_Per_Element
        return batch_size * self.hidden_size * self.dtype_bytes_per_elem


@dataclass(frozen=True)
class CommCostEstimate:
    measured_tp1_tok_s: float
    measured_tp2_tok_s: float
    observed_delta_percent: float
    inferred_alpha_us_per_step: float
    inferred_total_comm_latency_ms_per_tok: float
    is_comm_bound: bool
    explanation: str


class AlphaBetaCommModel:
    """
    Analytical Alpha-Beta Model: T_comm = N_steps * (alpha + S / beta)
    
    In PCIe PHB topologies (without NVLink), alpha represents the CPU root-complex 
    synchronization latency per collective call (~5-12 us).
    """

    # Model specs for key benchmark targets
    SPECS = {
        "opt-125m": ModelArchitectureSpecs(num_layers=12, hidden_size=768, num_attention_heads=12),
        "qwen2.5-3b": ModelArchitectureSpecs(num_layers=36, hidden_size=2048, num_attention_heads=16),
    }

    def __init__(self, default_alpha_us: float = 7.91, default_beta_gb_s: float = 7.8):
        """
        :param default_alpha_us: Inferred PCIe PHB step latency (~7.91 us per AllReduce step derived from M1)
        :param default_beta_gb_s: Effective PCIe Gen3 x8 bus bandwidth (~7.8 GB/s)
        """
        self.alpha_us = default_alpha_us
        self.beta_gb_s = default_beta_gb_s

    def estimate_comm_overhead_ms(self, model_key: str, tp_size: int = 2, batch_size: int = 1) -> float:
        """
        Estimates total inter-GPU communication latency per token in milliseconds.
        """
        if tp_size <= 1:
            return 0.0

        specs = self.SPECS.get(model_key.lower())
        if not specs:
            # Fallback architecture
            specs = ModelArchitectureSpecs(num_layers=24, hidden_size=2048, num_attention_heads=16)

        num_calls = specs.allreduce_calls_per_token
        s_bytes = specs.payload_bytes_per_allreduce(batch_size)
        
        # Ring AllReduce transfer penalty factor = 2 * (TP - 1) / TP
        tp_factor = 2.0 * (tp_size - 1) / tp_size

        # T_step = alpha + (tp_factor * S) / beta
        step_alpha_ms = self.alpha_us / 1000.0
        step_beta_ms = (tp_factor * s_bytes) / (self.beta_gb_s * 1e6)  # convert GB/s to bytes/ms
        
        single_step_comm_ms = step_alpha_ms + step_beta_ms
        return num_calls * single_step_comm_ms

    def evaluate_cell(
        self, 
        model_key: str, 
        tp1_tok_s: float, 
        tp2_tok_s: float, 
        concurrency: int = 1
    ) -> CommCostEstimate:
        """
        Compares observed TP1 vs TP2 tok/s against the analytical communication model.
        """
        if tp1_tok_s <= 0 or tp2_tok_s <= 0:
            return CommCostEstimate(
                measured_tp1_tok_s=tp1_tok_s,
                measured_tp2_tok_s=tp2_tok_s,
                observed_delta_percent=0.0,
                inferred_alpha_us_per_step=0.0,
                inferred_total_comm_latency_ms_per_tok=0.0,
                is_comm_bound=False,
                explanation="Invalid or missing throughput measurements."
            )

        delta_pct = ((tp2_tok_s - tp1_tok_s) / tp1_tok_s) * 100.0
        
        # Observed ms per token
        t_tp1_ms = 1000.0 / tp1_tok_s
        t_tp2_ms = 1000.0 / tp2_tok_s
        observed_delta_ms = t_tp2_ms - t_tp1_ms

        specs = self.SPECS.get(model_key.lower(), ModelArchitectureSpecs(num_layers=24, hidden_size=2048, num_attention_heads=16))
        num_calls = specs.allreduce_calls_per_token

        # Infer empirical alpha if TP2 is slower than TP1
        if observed_delta_ms > 0:
            inferred_alpha_us = (observed_delta_ms / num_calls) * 1000.0
            is_comm_bound = True
            explanation = (
                f"TP2 is {abs(delta_pct):.1f}% slower due to PCIe PHB bus latency. "
                f"Inferred AllReduce penalty is ~{inferred_alpha_us:.2f} µs per step across {num_calls} layers."
            )
        else:
            inferred_alpha_us = 0.0
            is_comm_bound = False
            explanation = (
                f"TP2 is {delta_pct:+.1f}% faster than TP1 at concurrency {concurrency}. "
                f"Arithmetic intensity (batch size={concurrency}) successfully hid PCIe communication latency."
            )

        est_comm_ms = self.estimate_comm_overhead_ms(model_key, tp_size=2, batch_size=concurrency)

        return CommCostEstimate(
            measured_tp1_tok_s=tp1_tok_s,
            measured_tp2_tok_s=tp2_tok_s,
            observed_delta_percent=delta_pct,
            inferred_alpha_us_per_step=inferred_alpha_us,
            inferred_total_comm_latency_ms_per_tok=est_comm_ms,
            is_comm_bound=is_comm_bound,
            explanation=explanation
        )
