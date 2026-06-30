"""Test CrossModalMerger (external bridge)."""
from fussion import merger


def test_merger_init():
    m = merger.CrossModalMerger(source_encoder="clip", target_llm="gpt2", bridge_type="mlp", device="cpu")
    assert m.llm_dim == 768
    assert m.source_encoder.name == "clip"
    assert m.bridge_type == "mlp"
