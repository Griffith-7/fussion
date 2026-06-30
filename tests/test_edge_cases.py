"""Edge case and boundary condition tests for fussion."""

import gc
import pytest
import torch

from fussion import bridge, datasets, encoders, fusion, merger
from fussion.exceptions import (
    BridgeNotFoundError,
    DimensionMismatchError,
    EncoderNotFoundError,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clean():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# BRIDGE — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestBridgeEdge:
    def test_zero_dims_vision_proj(self):
        with pytest.raises((ValueError, RuntimeError)):
            bridge.VisionProj(0, 128)
        with pytest.raises((ValueError, RuntimeError)):
            bridge.VisionProj(64, 0)

    def test_negative_dims_vision_proj(self):
        with pytest.raises((ValueError, RuntimeError)):
            bridge.VisionProj(-1, 128)

    def test_one_dim_vision_proj(self):
        b = bridge.VisionProj(1, 1)
        x = torch.randn(2, 10, 1)
        out = b(x)
        assert out.shape == (2, 10, 1)

    def test_asymmetric_dims_vision_proj(self):
        b = bridge.VisionProj(64, 128)
        x = torch.randn(2, 10, 64)
        out = b(x)
        assert out.shape == (2, 10, 128)

    def test_zero_dims_mlp_proj(self):
        with pytest.raises((ValueError, RuntimeError)):
            bridge.MLPVisionProj(0, 128)

    def test_one_dim_mlp_proj(self):
        b = bridge.MLPVisionProj(1, 1)
        x = torch.randn(2, 10, 1)
        out = b(x)
        assert out.shape == (2, 10, 1)

    def test_zero_dims_transformer_proj(self):
        with pytest.raises((ValueError, RuntimeError)):
            bridge.TransformerVisionProj(0, 128)

    def test_min_dims_transformer_proj(self):
        b = bridge.TransformerVisionProj(4, 4)
        x = torch.randn(2, 10, 4)
        out = b(x)
        assert out.shape == (2, 10, 4)

    def test_get_bridge_unknown_type(self):
        with pytest.raises(BridgeNotFoundError):
            bridge.get_bridge("nonexistent_bridge", 64, 128)

    def test_get_bridge_none_name(self):
        with pytest.raises(BridgeNotFoundError):
            bridge.get_bridge(None, 64, 128)  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# ENCODERS — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestEncoderEdge:
    def test_get_encoder_unknown_name(self):
        with pytest.raises(EncoderNotFoundError):
            encoders.get_encoder("nonexistent_encoder")

    def test_get_encoder_already_instance_passthrough(self):
        class FakeEncoder(encoders.ModalityEncoder):
            dim = 64
            def __init__(self):
                self.device = torch.device("cpu")
            def encode(self, x):
                return torch.randn(1, 64)
            def preprocess(self, x):
                return x

        inst = FakeEncoder()
        result = encoders.get_encoder(inst)
        assert result is inst

    def test_list_modalities(self):
        mods = encoders.list_modalities()
        assert isinstance(mods, list)
        assert "clip" in mods
        assert "video" in mods
        assert "codebert" in mods
        assert "text_llm" in mods


# ═══════════════════════════════════════════════════════════════════════════
# DATASETS — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetEdge:
    def test_n_zero_shape(self):
        (train_src, train_tgt), (val_src, val_tgt) = datasets.make_shape_dataset(n=0)
        assert len(train_src) == 0

    def test_n_one_shape(self):
        (train_src, train_tgt), (val_src, val_tgt) = datasets.make_shape_dataset(n=1)
        assert len(train_src) + len(val_src) >= 1

    def test_n_zero_text(self):
        (train_src, train_tgt), (val_src, val_tgt) = datasets.make_text_dataset(n=0)
        assert len(train_src) == 0

    def test_n_zero_code(self):
        (train_src, train_tgt), (val_src, val_tgt) = datasets.make_code_dataset(n=0)
        assert len(train_src) == 0

    def test_n_zero_video(self):
        (train_src, train_tgt), (val_src, val_tgt) = datasets.make_video_dataset(n=0)
        assert len(train_src) == 0

    def test_large_n_shape(self):
        (train_src, train_tgt), (val_src, val_tgt) = datasets.make_shape_dataset(n=100)
        total = len(train_src) + len(val_src)
        assert total >= 100

    def test_unknown_dataset_registry(self):
        with pytest.raises(KeyError):
            datasets.DATASET_REGISTRY["nonexistent_dataset"]


# ═══════════════════════════════════════════════════════════════════════════
# FUSION LLM — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestFusionLLMEdge:
    def test_invalid_encoder_dim_raises(self):
        with pytest.raises(DimensionMismatchError):
            fusion.FusionLLM(encoder_dim=-1, llm_name="distilgpt2")

    def test_zero_encoder_dim_raises(self):
        with pytest.raises(DimensionMismatchError):
            fusion.FusionLLM(encoder_dim=0, llm_name="distilgpt2")

    def test_unsupported_llm_name_raises(self):
        with pytest.raises((OSError, ValueError, Exception)):
            fusion.FusionLLM(encoder_dim=64, llm_name="this-model-does-not-exist-12345")

    def test_custom_kwargs(self):
        model = fusion.FusionLLM(
            encoder_dim=64,
            llm_name="distilgpt2",
            llm_kwargs={"attn_pdrop": 0.2, "embd_pdrop": 0.2},
        )
        assert model.llm.config.attn_pdrop == 0.2
        del model

    def test_every_k_layers_zero(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            fusion.FusionLLM(encoder_dim=64, llm_name="distilgpt2", every_k_layers=0)


# ═══════════════════════════════════════════════════════════════════════════
# MERGER — edge cases
# ═══════════════════════════════════════════════════════════════════════════

class TestMergerEdge:
    def test_invalid_bridge_type(self):
        with pytest.raises(BridgeNotFoundError):
            merger.CrossModalMerger(
                source_encoder="clip",
                target_llm="distilgpt2",
                bridge_type="nonexistent",
            )

    def test_evaluate_no_data(self):
        with torch.no_grad():
            m = merger.CrossModalMerger(
                source_encoder="clip",
                target_llm="distilgpt2",
            )
            eval_result = m.evaluate([], [])
            assert eval_result == float("inf")

    def test_save_and_load_no_config(self, tmp_path):
        with torch.no_grad():
            m = merger.CrossModalMerger(
                source_encoder="clip",
                target_llm="distilgpt2",
            )
            save_dir = str(tmp_path / "merger_test")
            m.save(save_dir)
            loaded = merger.CrossModalMerger(
                source_encoder="clip",
                target_llm="distilgpt2",
            )
            loaded.load(save_dir)
            assert loaded.llm_dim == m.llm_dim
            assert loaded.llm_name == m.llm_name
            del m, loaded
            clean()
