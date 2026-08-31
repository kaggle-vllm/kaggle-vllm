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


class ProfileError(KaggleVLLMError):
    """Raised when a packaged compatibility profile is missing or invalid."""


class DownloadError(KaggleVLLMError):
    """Raised when a pinned runtime artifact cannot be downloaded safely."""


class BootstrapError(KaggleVLLMError):
    """Raised when explicit native-runtime bootstrap cannot proceed safely."""


class BenchmarkError(KaggleVLLMError):
    """Raised for invalid, unsafe, or incompatible benchmark operations."""
