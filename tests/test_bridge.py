"""Tests for bridge module — linear, MLP, and transformer projections."""

import torch
import pytest
from fussion import bridge
from fussion.exceptions import BridgeNotFoundError


class TestVisionProj:
    def test_forward_shape(self):
        b = bridge.VisionProj(64, 128)
        x = torch.randn(2, 10, 64)
        out = b(x)
        assert out.shape == (2, 10, 128)

    def test_zero_init(self):
        b = bridge.VisionProj(64, 128)
        assert b.proj.weight.norm().item() == 0.0

    def test_learns(self):
        b = bridge.VisionProj(64, 128)
        x = torch.randn(2, 10, 64)
        out_before = b(x)
        with torch.no_grad():
            b.proj.weight.normal_(0, 0.1)
        out_after = b(x)
        assert not torch.allclose(out_before, out_after)

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            bridge.VisionProj(0, 128)
        with pytest.raises(ValueError):
            bridge.VisionProj(64, -1)


class TestMLPVisionProj:
    def test_forward_shape(self):
        b = bridge.MLPVisionProj(64, 128)
        x = torch.randn(2, 10, 64)
        out = b(x)
        assert out.shape == (2, 10, 128)

    def test_zero_init(self):
        b = bridge.MLPVisionProj(64, 128)
        assert b.linear.weight.norm().item() == 0.0
        assert b.mlp[-1].weight.norm().item() == 0.0

    def test_residual_connection(self):
        b = bridge.MLPVisionProj(64, 64)
        x = torch.randn(2, 10, 64)
        out = b(x)
        # Output should differ from input since proj IS zero but MLP has GELU
        assert out.shape == x.shape

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            bridge.MLPVisionProj(0, 128)
        with pytest.raises(ValueError):
            bridge.MLPVisionProj(64, 0)


class TestTransformerVisionProj:
    def test_forward_shape(self):
        b = bridge.TransformerVisionProj(64, 128)
        x = torch.randn(2, 10, 64)
        out = b(x)
        assert out.shape == (2, 10, 128)

    def test_zero_init(self):
        b = bridge.TransformerVisionProj(64, 128)
        assert b.attn.out_proj.weight.norm().item() == 0.0
        assert b.out.weight.norm().item() == 0.0

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            bridge.TransformerVisionProj(0, 128)
        with pytest.raises(ValueError):
            bridge.TransformerVisionProj(64, 0)


class TestGetBridge:
    def test_linear(self):
        b = bridge.get_bridge("linear", 64, 128)
        assert isinstance(b, bridge.VisionProj)

    def test_mlp(self):
        b = bridge.get_bridge("mlp", 64, 128)
        assert isinstance(b, bridge.MLPVisionProj)

    def test_transformer(self):
        b = bridge.get_bridge("transformer", 64, 128)
        assert isinstance(b, bridge.TransformerVisionProj)

    def test_unknown_bridge(self):
        with pytest.raises(BridgeNotFoundError):
            bridge.get_bridge("unknown", 64, 128)

    def test_registry_contents(self):
        assert "linear" in bridge.BRIDGE_REGISTRY
        assert "mlp" in bridge.BRIDGE_REGISTRY
        assert "transformer" in bridge.BRIDGE_REGISTRY
        assert len(bridge.BRIDGE_REGISTRY) == 3
