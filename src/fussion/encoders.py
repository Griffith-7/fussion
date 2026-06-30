"""Modality encoders: CLIP, Whisper, Video, CodeBERT, Text."""

import logging
from typing import Any, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .exceptions import EncoderError, EncoderNotFoundError

__all__ = [
    "ModalityEncoder",
    "ENCODER_REGISTRY",
    "get_encoder",
    "list_modalities",
    "CLIPEncoder",
    "WhisperEncoder",
    "VideoEncoder",
    "CodeEncoder",
    "TextEncoder",
]

logger = logging.getLogger(__name__)


class ModalityEncoder:
    """Base class for all modality encoders."""

    name: str = "base"
    dim: int = 512

    def __call__(self, inputs: Any) -> torch.Tensor:
        raise NotImplementedError

    def preprocess(self, inputs: Any) -> Any:
        return inputs


ENCODER_REGISTRY: dict = {}


def register(cls):
    ENCODER_REGISTRY[cls.name] = cls
    return cls


def get_encoder(name: Union[str, ModalityEncoder], device: Optional[torch.device] = None) -> ModalityEncoder:
    if isinstance(name, ModalityEncoder):
        return name
    if name in ENCODER_REGISTRY:
        logger.debug("Creating encoder '%s' on device %s", name, device or "auto")
        return ENCODER_REGISTRY[name](device=device)
    raise EncoderNotFoundError(
        f"Unknown encoder: '{name}'. Options: {list(ENCODER_REGISTRY)}"
    )


def list_modalities() -> List[str]:
    return list(ENCODER_REGISTRY)


@register
class CLIPEncoder(ModalityEncoder):
    name = "clip"
    dim = 512

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            import clip
        except ImportError as e:
            raise EncoderError("CLIP is not installed. Run: pip install openai-clip") from e
        self.model, self.preprocessor = clip.load(model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, inputs: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], Image.Image):
            images = [self.preprocessor(img).unsqueeze(0) for img in inputs]
            images = torch.cat(images, dim=0).to(self.device)
            return self.model.encode_image(images).unsqueeze(1)
        return self.model.encode_image(self.preprocessor(inputs).unsqueeze(0).to(self.device)).unsqueeze(1)


@register
class WhisperEncoder(ModalityEncoder):
    name = "whisper"
    dim = 512

    def __init__(self, model_name: str = "small", device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            import whisper
        except ImportError as e:
            raise EncoderError("Whisper is not installed. Run: pip install openai-whisper") from e
        self.model = whisper.load_model(model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, inputs: Union[np.ndarray, List[np.ndarray], str]) -> torch.Tensor:
        import numpy as np
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], np.ndarray):
            out = []
            for audio in inputs:
                audio_tensor = torch.tensor(audio, dtype=torch.float32).to(self.device)
                mel = self.model.mel_spectrogram(audio_tensor.unsqueeze(0))
                features = self.model.encoder(mel)
                out.append(features.mean(dim=1, keepdim=True))
            return torch.cat(out, dim=0)
        if isinstance(inputs, np.ndarray):
            audio_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            mel = self.model.mel_spectrogram(audio_tensor.unsqueeze(0))
            return self.model.encoder(mel).mean(dim=1, keepdim=True)
        if isinstance(inputs, str):
            self.model.transcribe(inputs)
            return torch.zeros(1, 1, self.dim, device=self.device)
        raise EncoderError(f"Unsupported input type for WhisperEncoder: {type(inputs)}")


@register
class VideoEncoder(ModalityEncoder):
    name = "video"
    dim = 512

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[torch.device] = None,
                 fps: int = 4, max_frames: int = 8) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = fps
        self.max_frames = max_frames
        self.clip_enc = CLIPEncoder(model_name, device)

    @torch.no_grad()
    def __call__(self, inputs: Union[List, List[List]]) -> torch.Tensor:
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], list):
            return self._encode_batch(inputs)
        return self._encode_single(inputs)

    def _encode_single(self, frames: List) -> torch.Tensor:
        frames = frames[:self.max_frames] if isinstance(frames, list) else []
        if not frames:
            return torch.zeros(1, 1, self.dim, device=self.device)
        frame_tokens = self.clip_enc(frames)
        return frame_tokens.mean(dim=0, keepdim=True)

    def _encode_batch(self, batch: List[List]) -> torch.Tensor:
        return torch.cat([self._encode_single(frames) for frames in batch], dim=0)


@register
class CodeEncoder(ModalityEncoder):
    name = "codebert"
    dim = 768

    def __init__(self, model_name: str = "microsoft/codebert-base", device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise EncoderError("transformers is not installed") from e
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(inputs, list):
            encodings = self.tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
            out = self.model(**encodings.to(self.device))
            return out.last_hidden_state
        encoding = self.tok(inputs, return_tensors="pt", truncation=True, max_length=128)
        out = self.model(**encoding.to(self.device))
        return out.last_hidden_state


@register
class TextEncoder(ModalityEncoder):
    name = "text_llm"
    dim = 768

    def __init__(self, model_name: str = "gpt2", device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise EncoderError("transformers is not installed") from e
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token or self.tok.pad_token
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, inputs: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(inputs, list):
            encodings = self.tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
            out = self.model(**encodings.to(self.device))
            return out.last_hidden_state
        encoding = self.tok(inputs, return_tensors="pt", truncation=True, max_length=128)
        out = self.model(**encoding.to(self.device))
        return out.last_hidden_state
