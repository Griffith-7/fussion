"""Generate images → text using a trained fusion model."""
from fussion import FusionLLM, train_fusion
from fussion.encoders import CLIPEncoder
from fussion.datasets import make_shape_dataset
from PIL import Image

# Train on synthetic shapes
(train_src, train_tgt), _ = make_shape_dataset(n=80)
encoder = CLIPEncoder()
fusion = FusionLLM("gpt2", encoder_dim=512, every_k_layers=4)
train_fusion(fusion, encoder, train_src, train_tgt, steps=15, batch_size=4)

# Generate on a NEW image (not from training set)
from PIL import ImageDraw
img = Image.new("RGB", (224, 224), (255, 255, 255))
ImageDraw.Draw(img).ellipse([52, 52, 172, 172], fill=(255, 0, 0))  # red circle

vis = encoder([img])
print("Generated caption:", fusion.generate(vis, temperature=0.5))
