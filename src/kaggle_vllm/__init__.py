"""Kaggle compatibility helpers around upstream vLLM."""

from .l4_bootstrap import activate_runtime, bootstrap
from .llm import KaggleLLM
from .profile_v3 import L4Profile, PROFILE_NAME, load_profile as _load_l4_profile
from .profiles import DEFAULT_PROFILE as T4_PROFILE
from .sampling import KaggleSamplingParams
from .gemma import discover_gemma4_qat
from .sharding import ShardedModelInspection, inspect_sharded_model

__all__ = [
    "L4Profile",
    "KaggleLLM",
    "KaggleSamplingParams",
    "ShardedModelInspection",
    "activate_runtime",
    "bootstrap",
    "discover_gemma4_qat",
    "inspect_sharded_model",
    "load_profile",
]
__version__ = "0.3.0"


def load_profile(profile_name: str = PROFILE_NAME):
    """Load either preserved T4 or new L4 compatibility metadata."""
    if profile_name == T4_PROFILE:
        from .profiles import load_profile as load_t4_profile
        return load_t4_profile(profile_name)
    return _load_l4_profile(profile_name)
