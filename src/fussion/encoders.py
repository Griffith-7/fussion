"""Modality encoders: CLIP, Whisper, Video, CodeBERT, Text."""
import torch
import torch.nn as nn
from typing import Any, List, Optional, Union
from PIL import Image


class ModalityEncoder:
    name = "base"
    dim = 512
    def __call__(self, inputs): raise NotImplementedError
    def preprocess(self, inputs): return inputs


ENCODER_REGISTRY = {}


def register(cls):
    ENCODER_REGISTRY[cls.name] = cls
    return cls


def get_encoder(name, device=None):
    if isinstance(name, ModalityEncoder): return name
    if name in ENCODER_REGISTRY:
        return ENCODER_REGISTRY[name](device=device)
    raise ValueError(f"Unknown encoder: {name}. Options: {list(ENCODER_REGISTRY)}")


def list_modalities(): return list(ENCODER_REGISTRY)


@register
class CLIPEncoder(ModalityEncoder):
    name = "clip"
    dim = 512
    def __init__(self, model_name="ViT-B/32", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import clip
        self.model, self.preprocessor = clip.load(model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, inputs):
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], Image.Image):
            images = [self.preprocessor(img).unsqueeze(0) for img in inputs]
            images = torch.cat(images, dim=0).to(self.device)
            return self.model.encode_image(images).unsqueeze(1)  # [B, 1, 512]
        return self.model.encode_image(self.preprocessor(inputs).unsqueeze(0).to(self.device)).unsqueeze(1)


@register
class WhisperEncoder(ModalityEncoder):
    name = "whisper"
    dim = 512  # whisper-small
    def __init__(self, model_name="small", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import whisper
        self.model = whisper.load_model(model_name, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, inputs):
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
            result = self.model.transcribe(inputs)
            return torch.zeros(1, 1, self.dim, device=self.device)
        raise TypeError(f"Unsupported input type: {type(inputs)}")


@register
class VideoEncoder(ModalityEncoder):
    name = "video"
    dim = 512
    def __init__(self, model_name="ViT-B/32", device=None, fps=4, max_frames=8):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = fps
        self.max_frames = max_frames
        from .encoders import CLIPEncoder
        self.clip_enc = CLIPEncoder(model_name, device)

    @torch.no_grad()
    def __call__(self, inputs):
        if isinstance(inputs, list) and inputs and isinstance(inputs[0], list):
            return self._encode_batch(inputs)
        return self._encode_single(inputs)

    def _encode_single(self, frames):
        frames = frames[:self.max_frames] if isinstance(frames, list) else []
        if not frames:
            return torch.zeros(1, 1, self.dim, device=self.device)
        frame_tokens = self.clip_enc(frames)  # [F, 1, 512]
        return frame_tokens.mean(dim=0, keepdim=True)  # [1, 1, 512]

    def _encode_batch(self, batch):
        return torch.cat([self._encode_single(frames) for frames in batch], dim=0)


@register
class CodeEncoder(ModalityEncoder):
    name = "codebert"
    dim = 768
    def __init__(self, model_name="microsoft/codebert-base", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, inputs):
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
    def __init__(self, model_name="gpt2", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token or self.tok.pad_token
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, inputs):
        if isinstance(inputs, list):
            encodings = self.tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
            out = self.model(**encodings.to(self.device))
            return out.last_hidden_state
        encoding = self.tok(inputs, return_tensors="pt", truncation=True, max_length=128)
        out = self.model(**encoding.to(self.device))
        return out.last_hidden_state
