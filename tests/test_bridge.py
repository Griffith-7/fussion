"""Tests for bridge architectures."""
import torch
from fussion import bridge


def test_linear():
    b = bridge.get_bridge("linear", 512, 768)
    y = b(torch.randn(2, 10, 512))
    assert y.shape == (2, 10, 768)


def test_mlp():
    b = bridge.get_bridge("mlp", 512, 768)
    y = b(torch.randn(2, 10, 512))
    assert y.shape == (2, 10, 768)


def test_transformer():
    b = bridge.get_bridge("transformer", 512, 768)
    y = b(torch.randn(2, 10, 512))
    assert y.shape == (2, 10, 768)


def test_zero_init():
    b = bridge.get_bridge("linear", 512, 768)
    assert b.proj.weight.sum().item() == 0.0
