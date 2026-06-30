"""Quickstart: train a bridge from CLIP → GPT-2."""
from fussion import CrossModalMerger
from fussion.datasets import make_shape_dataset

# Generate synthetic data
(train_src, train_tgt), (val_src, val_tgt) = make_shape_dataset(n=80)

# External bridge (modular, lightweight)
merger = CrossModalMerger(source_encoder="clip", target_llm="gpt2", bridge_type="mlp")
merger.train_bridge(train_src, train_tgt, val_src, val_tgt, steps=20, batch_size=8)

print(f"Val PPL: {merger.evaluate(val_src, val_tgt):.2f}")
print("Generated:", merger.generate(val_src[0]))
