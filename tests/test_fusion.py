"""Tests for fusion model."""
import torch
from fussion import fusion


def test_fusion_init():
    f = fusion.FusionLLM("gpt2", encoder_dim=512, every_k_layers=4, device="cpu", verbose=False)
    params = sum(p.numel() for p in f.get_trainable_params())
    assert params > 0
    assert len(f.wrapped_layers) > 0


def test_fusion_forward():
    f = fusion.FusionLLM("gpt2", encoder_dim=512, every_k_layers=4, device="cpu", verbose=False)
    B, N, D_enc, T = 2, 50, 512, 10
    vis = torch.randn(B, N, D_enc)
    ids = torch.randint(0, 100, (B, T))
    out = f(ids, vis)
    assert "logits" in out
    assert out["logits"].shape[0] == B
    assert out["logits"].shape[-1] == f.vocab_size


def test_fusion_loss():
    f = fusion.FusionLLM("gpt2", encoder_dim=512, every_k_layers=4, device="cpu", verbose=False)
    B, N, D_enc, T = 2, 50, 512, 10
    vis = torch.randn(B, N, D_enc)
    ids = torch.randint(0, f.vocab_size - 1, (B, T + 1))
    labels = ids[:, 1:].clone()
    out = f(ids[:, :-1], vis, labels=labels)
    assert "loss" in out
    assert out["loss"].item() > 0
