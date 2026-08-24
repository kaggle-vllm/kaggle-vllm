"""Kaggle compatibility helpers around upstream vLLM."""

from .llm import KaggleLLM
from .sharding import ShardedModelInspection, inspect_sharded_model

__all__ = ["KaggleLLM", "ShardedModelInspection", "inspect_sharded_model"]
__version__ = "0.1.0"
