"""fussion — Cross-modal fusion: connect any encoder to any frozen LLM."""

import logging

from .bridge import get_bridge
from .merger import CrossModalMerger
from .fusion import FusionLLM, train_fusion
from .encoders import (
    ModalityEncoder, CLIPEncoder, WhisperEncoder,
    VideoEncoder, CodeEncoder, TextEncoder,
    get_encoder, list_modalities,
)
from .exceptions import (
    FussionError,
    EncoderNotFoundError,
    BridgeNotFoundError,
    DimensionMismatchError,
    EncoderError,
)

__version__ = "0.2.0"


__all__ = [
    "CrossModalMerger",
    "FusionLLM",
    "train_fusion",
    "get_bridge",
    "get_encoder",
    "list_modalities",
    "ModalityEncoder",
    "CLIPEncoder",
    "WhisperEncoder",
    "VideoEncoder",
    "CodeEncoder",
    "TextEncoder",
    "FussionError",
    "EncoderNotFoundError",
    "BridgeNotFoundError",
    "DimensionMismatchError",
    "EncoderError",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
