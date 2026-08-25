"""Kaggle compatibility helpers around upstream vLLM."""

from .bootstrap import activate_runtime, bootstrap
from .llm import KaggleLLM
from .profiles import BootstrapProfile, load_profile
from .sharding import ShardedModelInspection, inspect_sharded_model

__all__ = [
    "BootstrapProfile",
    "KaggleLLM",
    "ShardedModelInspection",
    "activate_runtime",
    "bootstrap",
    "inspect_sharded_model",
    "load_profile",
]
__version__ = "0.1.2"
