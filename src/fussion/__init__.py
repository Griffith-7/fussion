"""fussion — Cross-modal fusion: connect any encoder to any frozen LLM."""

import logging

from .bridge import get_bridge
from .encoders import (
    CLIPEncoder,
    CodeEncoder,
    ModalityEncoder,
    TextEncoder,
    VideoEncoder,
    WhisperEncoder,
    get_encoder,
    list_modalities,
)
from .exceptions import (
    BridgeNotFoundError,
    DimensionMismatchError,
    EncoderError,
    EncoderNotFoundError,
    FussionError,
)
from .fusion import FusionLLM, train_fusion
from .merger import CrossModalMerger

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
