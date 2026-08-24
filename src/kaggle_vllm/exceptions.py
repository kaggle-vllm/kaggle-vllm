"""Public exception hierarchy for kaggle-vllm."""


class KaggleVLLMError(RuntimeError):
    """Base exception for SDK-specific failures."""


class RuntimeValidationError(KaggleVLLMError):
    """Raised when visible hardware cannot satisfy a request."""


class VLLMNotInstalledError(KaggleVLLMError):
    """Raised when an inference operation needs an unavailable vLLM install."""


class ChecksumMismatchError(KaggleVLLMError):
    """Raised when an artifact does not match its expected SHA256."""


class ShardedModelError(KaggleVLLMError):
    """Raised when a persistent sharded-state directory is invalid."""


class InstallationError(KaggleVLLMError):
    """Raised for unsafe or failed explicit staging operations."""
