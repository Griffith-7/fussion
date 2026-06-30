"""Tests for modality encoders."""
from fussion import encoders


def test_registry():
    names = encoders.list_modalities()
    assert "clip" in names
    assert "whisper" in names
    assert "video" in names
    assert "codebert" in names
    assert "text_llm" in names


def test_cls_attributes():
    for name in encoders.list_modalities():
        cls = encoders.ENCODER_REGISTRY[name]
        assert hasattr(cls, "name")
        assert hasattr(cls, "dim")
        assert callable(cls)


def test_cls_consistency():
    """Verify all registered names match their cls.name."""
    for name, cls in encoders.ENCODER_REGISTRY.items():
        assert cls.name == name, f"{cls.__name__}.name='{cls.name}' != registry key '{name}'"
