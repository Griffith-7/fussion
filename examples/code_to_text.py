"""Swap encoders: use CodeBERT instead of CLIP with the same API."""
from fussion import CrossModalMerger
from fussion.datasets import make_code_dataset

(train_src, train_tgt), (val_src, val_tgt) = make_code_dataset(n=40)

merger = CrossModalMerger(source_encoder="codebert", target_llm="gpt2", bridge_type="mlp")
merger.train_bridge(train_src, train_tgt, val_src, val_tgt, steps=15, batch_size=4)

print(f"Code → text PPL: {merger.evaluate(val_src, val_tgt):.2f}")
print("Generated:", merger.generate(val_src[0]))
