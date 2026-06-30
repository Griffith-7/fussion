"""Tests for FusionLLM — init, forward, generation, and error paths."""

import torch
import pytest
from fussion import fusion
from fussion.exceptions import DimensionMismatchError


class TestFusionLLMInit:
    def test_default_init(self):
        f = fusion.FusionLLM("distilgpt2", encoder_dim=512, device="cpu")
        assert f.d_model == 768
        assert f.llm_name == "distilgpt2"
        assert hasattr(f, "wrapped_layers")
        assert len(f.wrapped_layers) > 0
        assert f.visual_proj.in_features == 512

    def test_trainable_params(self):
        f = fusion.FusionLLM("distilgpt2", encoder_dim=512, device="cpu")
        params = f.get_trainable_params()
        assert len(params) > 0
        for p in params:
            assert p.requires_grad

    def test_every_k_layers(self):
        f = fusion.FusionLLM("distilgpt2", encoder_dim=512, every_k_layers=2, device="cpu")
        n_fusion = sum(1 for wl in f.wrapped_layers if wl.cross_attn is not None)
        assert n_fusion > 0
        # With 6 layers (distilgpt2) and every_k=2, we expect 3 fusion layers
        assert n_fusion == 3

    def test_invalid_encoder_dim(self):
        with pytest.raises(DimensionMismatchError):
            fusion.FusionLLM("distilgpt2", encoder_dim=0, device="cpu")

    def test_custom_llm_kwargs(self):
        f = fusion.FusionLLM("distilgpt2", encoder_dim=512, device="cpu",
                              llm_kwargs={"dtype": torch.float32})
        assert f.llm_dtype == torch.float32


class TestFusionLLMForward:
    def test_forward_logits(self):
        f = fusion.FusionLLM("distilgpt2", encoder_dim=512, device="cpu")
        visual = torch.randn(2, 4, 512)
        ids = torch.randint(0, 100, (2, 8))
        out = f(ids, visual)
        assert "logits" in out
        assert out["logits"].shape[0] == 2

    def test_forward_with_loss(self):
        f = fusion.FusionLLM("distilgpt2", encoder_dim=512, device="cpu")
        visual = torch.randn(2, 4, 512)
        ids = torch.randint(0, 100, (2, 8))
        labels = ids.clone()
        out = f(ids, visual, labels=labels)
        assert "loss" in out
        assert out["loss"].item() > 0

    def test_forward_mask(self):
        f = fusion.FusionLLM("distilgpt2", encoder_dim=512, device="cpu")
        visual = torch.randn(2, 4, 512)
        ids = torch.randint(0, 100, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)
        out = f(ids, visual, attention_mask=mask)
        assert "logits" in out


class TestFusionLLMGeneration:
    def test_generate_empty_prompt(self):
        f = fusion.FusionLLM("distilgpt2", encoder_dim=512, device="cpu")
        visual = torch.randn(1, 4, 512)
        text = f.generate(visual, max_new=5)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_with_prompt(self):
        f = fusion.FusionLLM("distilgpt2", encoder_dim=512, device="cpu")
        visual = torch.randn(1, 4, 512)
        text = f.generate(visual, prompt="Hello", max_new=5)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_generate_temperature_zero(self):
        f = fusion.FusionLLM("distilgpt2", encoder_dim=512, device="cpu")
        visual = torch.randn(1, 4, 512)
        text = f.generate(visual, max_new=5, temperature=0)
        assert isinstance(text, str)
