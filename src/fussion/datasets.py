"""Synthetic and real dataset generators for benchmarking."""

import logging
import random
from typing import Callable, Dict, List, Tuple

from PIL import Image, ImageDraw

__all__ = [
    "make_shape_dataset",
    "make_video_dataset",
    "make_code_dataset",
    "make_text_dataset",
    "DATASET_REGISTRY",
]

logger = logging.getLogger(__name__)

COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (255, 50, 50),
    "blue": (50, 50, 255),
    "green": (50, 200, 50),
}
SHAPES: List[str] = ["circle", "square", "triangle"]

DatasetSplit = Tuple[List, List]


def make_shape_dataset(n: int = 100) -> Tuple[DatasetSplit, DatasetSplit]:
    """Simple color+shape dataset: 'a {color} {shape}'."""
    src, tgt = [], []
    for _ in range(n):
        c = random.choice(list(COLORS.keys()))
        s = random.choice(SHAPES)
        img = Image.new("RGB", (224, 224), (255, 255, 255))
        d = ImageDraw.Draw(img)
        r, cx, cy = 60, 112, 112
        if s == "circle":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLORS[c])
        elif s == "square":
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=COLORS[c])
        else:
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=COLORS[c])
        src.append(img)
        tgt.append(f"a {c} {s}")
    split = int(n * 0.8)
    return (src[:split], tgt[:split]), (src[split:], tgt[split:])


def make_video_dataset(n: int = 40) -> Tuple[DatasetSplit, DatasetSplit]:
    """Simple moving circle video dataset."""
    colors = {"red": (255, 0, 0), "blue": (0, 0, 255)}
    src, tgt = [], []
    for _ in range(n):
        c = random.choice(list(colors.keys()))
        frames = [Image.new("RGB", (224, 224), (255, 255, 255)) for _ in range(5)]
        for fi, f in enumerate(frames):
            ImageDraw.Draw(f).ellipse([82 + fi * 5, 82, 142 + fi * 5, 142], fill=colors[c])
        src.append(frames)
        tgt.append(f"a {c} moving circle")
    split = int(n * 0.8)
    return (src[:split], tgt[:split]), (src[split:], tgt[split:])


def make_code_dataset(n: int = 40) -> Tuple[DatasetSplit, DatasetSplit]:
    """Code snippet dataset."""
    src, tgt = [], []
    for _ in range(n):
        lang = random.choice(["python", "javascript", "rust"])
        code = {
            "python": "def hello():\n    print('hi')",
            "javascript": "function hello() { console.log('hi'); }",
            "rust": "fn hello() { println!(\"hi\"); }",
        }[lang]
        src.append(code)
        tgt.append(f"a {lang} function")
    split = int(n * 0.8)
    return (src[:split], tgt[:split]), (src[split:], tgt[split:])


def make_text_dataset(n: int = 40) -> Tuple[DatasetSplit, DatasetSplit]:
    """Text-to-text dataset."""
    topics = ["science", "art", "music"]
    src = [f"Tell me about {t}" for t in topics]
    tgt = [f"{t} is a fascinating field of study" for t in topics]
    src = (src * (n // len(src) + 1))[:n]
    tgt = (tgt * (n // len(tgt) + 1))[:n]
    split = int(n * 0.8)
    return (src[:split], tgt[:split]), (src[split:], tgt[split:])


DATASET_REGISTRY: Dict[str, Callable] = {
    "shape": make_shape_dataset,
    "video": make_video_dataset,
    "code": make_code_dataset,
    "text": make_text_dataset,
}
