"""Custom exceptions for fussion."""

__all__ = [
    "FussionError",
    "EncoderNotFoundError",
    "BridgeNotFoundError",
    "DimensionMismatchError",
    "EncoderError",
]


class FussionError(Exception):
    """Base exception for all fussion errors."""


class EncoderNotFoundError(FussionError, ValueError):
    """Raised when an encoder is not found in the registry."""


class BridgeNotFoundError(FussionError, ValueError):
    """Raised when a bridge type is not found in the registry."""


class DimensionMismatchError(FussionError, ValueError):
    """Raised when dimensions between components are incompatible."""


class EncoderError(FussionError, RuntimeError):
    """Raised when an encoder fails during forward pass."""
