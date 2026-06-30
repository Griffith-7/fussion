"""Tests for datasets."""
from fussion import datasets


def test_shape_dataset():
    (train_src, train_tgt), (val_src, val_tgt) = datasets.make_shape_dataset(n=20)
    assert len(train_src) > 0
    assert len(train_tgt) > 0
    assert len(train_src) == len(train_tgt)


def test_video_dataset():
    (train_src, train_tgt), (val_src, val_tgt) = datasets.make_video_dataset(n=10)
    assert len(train_src) > 0
    assert len(train_src) == len(train_tgt)


def test_code_dataset():
    (train_src, train_tgt), (val_src, val_tgt) = datasets.make_code_dataset(n=10)
    assert len(train_src) == len(train_tgt)


def test_text_dataset():
    (train_src, train_tgt), (val_src, val_tgt) = datasets.make_text_dataset(n=10)
    assert len(train_src) == len(train_tgt)


def test_registry():
    assert "shape" in datasets.DATASET_REGISTRY
    assert "video" in datasets.DATASET_REGISTRY
    assert "code" in datasets.DATASET_REGISTRY
    assert "text" in datasets.DATASET_REGISTRY
