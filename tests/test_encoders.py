"""Tests for encoders module — registry, class attributes, and error paths."""

import pytest
from fussion import encoders
from fussion.exceptions import EncoderNotFoundError


class TestEncoderRegistry:
    def test_all_modalities_present(self):
        names = encoders.list_modalities()
        assert "clip" in names
        assert "whisper" in names
        assert "video" in names
        assert "codebert" in names
        assert "text_llm" in names
        assert len(names) == 5

    def test_registry_classes_have_attributes(self):
        for cls in encoders.ENCODER_REGISTRY.values():
            assert hasattr(cls, "name")
            assert hasattr(cls, "dim")
            assert hasattr(cls, "__call__")

    def test_registry_keys_match_names(self):
        for key, cls in encoders.ENCODER_REGISTRY.items():
            assert cls.name == key

    def test_clip_dim(self):
        assert encoders.CLIPEncoder.dim == 512

    def test_codebert_dim(self):
        assert encoders.CodeEncoder.dim == 768

    def test_text_llm_dim(self):
        assert encoders.TextEncoder.dim == 768


class TestGetEncoder:
    def test_get_by_name(self):
        enc = encoders.get_encoder("clip")
        assert isinstance(enc, encoders.CLIPEncoder)

    def test_get_already_instance(self):
        enc = encoders.CLIPEncoder(device="cpu")
        same = encoders.get_encoder(enc)
        assert same is enc

    def test_unknown_encoder(self):
        with pytest.raises(EncoderNotFoundError):
            encoders.get_encoder("nonexistent_encoder")


class TestBaseEncoder:
    def test_base_raises_not_implemented(self):
        base = encoders.ModalityEncoder()
        with pytest.raises(NotImplementedError):
            base("test")

    def test_preprocess_identity(self):
        base = encoders.ModalityEncoder()
        assert base.preprocess(42) == 42
