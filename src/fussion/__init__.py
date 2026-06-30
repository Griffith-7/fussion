"""fussion — Cross-modal fusion: connect any encoder to any frozen LLM."""

from .bridge import get_bridge
from .merger import CrossModalMerger
from .fusion import FusionLLM, train_fusion
from .encoders import (
    ModalityEncoder, CLIPEncoder, WhisperEncoder,
    VideoEncoder, CodeEncoder, TextEncoder,
    get_encoder, list_modalities,
)

__version__ = "0.1.0"
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
]
