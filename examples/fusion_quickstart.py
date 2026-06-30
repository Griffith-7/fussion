"""Quickstart: train fusion (cross-attention) from CLIP → GPT-2."""
from fussion import FusionLLM, train_fusion
from fussion.encoders import CLIPEncoder
from fussion.datasets import make_shape_dataset

# Generate synthetic data
(train_src, train_tgt), (val_src, val_tgt) = make_shape_dataset(n=80)

encoder = CLIPEncoder()
fusion = FusionLLM("gpt2", encoder_dim=512, every_k_layers=4)

train_fusion(fusion, encoder, train_src, train_tgt, val_src, val_tgt, steps=20, batch_size=4)

vis = encoder([val_src[0]])
print("Generated:", fusion.generate(vis))
