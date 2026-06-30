"""Tests for CrossModalMerger — init, save/load, evaluation, and error paths."""

import os
import torch
import pytest
from fussion import merger
from fussion.exceptions import BridgeNotFoundError


class TestCrossModalMergerInit:
    def test_init_gpt2(self):
        m = merger.CrossModalMerger("clip", "gpt2", device="cpu")
        assert m.llm_dim == 768
        assert m.source_encoder.name == "clip"
        assert m.bridge_type == "mlp"
        assert m.bridge is not None

    def test_init_distilgpt2(self):
        m = merger.CrossModalMerger("clip", "distilgpt2", device="cpu")
        assert m.llm_dim == 768
        assert m.source_encoder.name == "clip"

    def test_bridge_has_correct_dims(self):
        m = merger.CrossModalMerger("clip", "gpt2", bridge_type="linear", device="cpu")
        assert m.bridge.proj.in_features == 512
        assert m.bridge.proj.out_features == m.llm_dim


class TestCrossModalMergerSaveLoad:
    def test_save_and_load(self, tmp_path):
        m = merger.CrossModalMerger("clip", "gpt2", bridge_type="linear", device="cpu")
        save_dir = str(tmp_path / "test_merger")
        m.save(save_dir)
        assert os.path.exists(os.path.join(save_dir, "bridge.pt"))
        assert os.path.exists(os.path.join(save_dir, "config.json"))

        # Load into a new merger
        m2 = merger.CrossModalMerger("clip", "gpt2", bridge_type="linear", device="cpu")
        m2.load(save_dir)
        # Weights should be identical (both zero-init)
        for p1, p2 in zip(m.bridge.parameters(), m2.bridge.parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    def test_load_with_different_bridge_type(self, tmp_path):
        m = merger.CrossModalMerger("clip", "gpt2", bridge_type="mlp", device="cpu")
        save_dir = str(tmp_path / "test_merger2")
        m.save(save_dir)

        m2 = merger.CrossModalMerger("clip", "gpt2", bridge_type="linear", device="cpu")
        m2.load(save_dir)
        # Should switch to MLP since config says mlp
        assert m2.bridge_type == "mlp"

    def test_save_load_no_config(self, tmp_path):
        m = merger.CrossModalMerger("clip", "gpt2", bridge_type="linear", device="cpu")
        save_dir = str(tmp_path / "test_merger3")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(m.bridge.state_dict(), os.path.join(save_dir, "bridge.pt"))
        # No config.json — should still load
        m2 = merger.CrossModalMerger("clip", "gpt2", bridge_type="linear", device="cpu")
        m2.load(save_dir)
        assert m2.bridge is not None

    def test_bridge_learns(self):
        m = merger.CrossModalMerger("clip", "gpt2", bridge_type="linear", device="cpu")
        x = torch.randn(1, 4, 512)
        out_before = m.bridge(x)
        with torch.no_grad():
            m.bridge.proj.weight.normal_(0, 0.1)
        out_after = m.bridge(x)
        assert not torch.allclose(out_before, out_after)

    def test_evaluate_returns_inf_on_empty(self):
        m = merger.CrossModalMerger("clip", "gpt2", bridge_type="linear", device="cpu")
        ppl = m.evaluate([], [])
        assert ppl == float("inf")
