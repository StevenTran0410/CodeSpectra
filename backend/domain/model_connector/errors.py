"""Normalized errors for the model connector layer."""
from enum import Enum


class ProviderErrorCode(str, Enum):
    CONNECTION_REFUSED = "connection_refused"
    AUTH_FAILED = "auth_failed"
    MODEL_NOT_FOUND = "model_not_found"
    CONTEXT_LIMIT_EXCEEDED = "context_limit_exceeded"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ProviderError(Exception):
    def __init__(
        self,
        code: ProviderErrorCode,
        message: str,
        provider_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.provider_id = provider_id
        self.retryable = retryable

    def __repr__(self) -> str:
        return f"ProviderError(code={self.code}, provider={self.provider_id}, message={self.message!r})"
