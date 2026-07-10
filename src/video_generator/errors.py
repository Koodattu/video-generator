from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorKind(StrEnum):
    NOT_READY = "not_ready"
    UNSUPPORTED = "unsupported"
    TRANSIENT = "transient"
    INVALID_OUTPUT = "invalid_output"
    POLICY_REFUSAL = "policy_refusal"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERNAL = "internal"


class VideoGeneratorError(Exception):
    """Base error with a stable category and actionable user message."""

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind = ErrorKind.INTERNAL,
        action: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.action = action
        self.details = details or {}


class ConfigurationError(VideoGeneratorError):
    def __init__(self, message: str, *, action: str | None = None) -> None:
        super().__init__(message, kind=ErrorKind.UNSUPPORTED, action=action)


class BackendError(VideoGeneratorError):
    pass


class CheckpointError(VideoGeneratorError):
    pass


class MediaError(VideoGeneratorError):
    pass

