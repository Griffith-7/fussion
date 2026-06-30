"""Tests for datasets module."""

import pytest
from fussion import datasets


class TestShapeDataset:
    def test_returns_splits(self):
        (s, t), (vs, vt) = datasets.make_shape_dataset(100)
        assert len(s) == 80
        assert len(t) == 80
        assert len(vs) == 20
        assert len(vt) == 20

    def test_image_and_text_pairs(self):
        (s, t), _ = datasets.make_shape_dataset(10)
        for img, txt in zip(s, t):
            assert txt.startswith("a ")
            assert any(c in txt for c in ["red", "blue", "green"])
            assert any(sh in txt for sh in ["circle", "square", "triangle"])

    def test_small_n(self):
        (s, t), (vs, vt) = datasets.make_shape_dataset(5)
        assert len(s) == 4
        assert len(vs) == 1


class TestVideoDataset:
    def test_returns_splits(self):
        (s, t), (vs, vt) = datasets.make_video_dataset(40)
        assert len(s) == 32
        assert len(vt) == 8

    def test_frames_are_lists(self):
        (s, t), _ = datasets.make_video_dataset(5)
        for frames in s:
            assert isinstance(frames, list)
            assert len(frames) > 0


class TestCodeDataset:
    def test_returns_splits(self):
        (s, t), _ = datasets.make_code_dataset(10)
        assert len(s) == 8

    def test_code_content(self):
        (s, t), _ = datasets.make_code_dataset(5)
        for code in s:
            assert isinstance(code, str)


class TestTextDataset:
    def test_returns_splits(self):
        (s, t), _ = datasets.make_text_dataset(10)
        assert len(s) == 8

    def test_topics_match(self):
        (s, t), _ = datasets.make_text_dataset(6)
        for src, tgt in zip(s, t):
            assert src.startswith("Tell me about")
            assert tgt.endswith("field of study")


class TestRegistry:
    def test_all_datasets_present(self):
        assert "shape" in datasets.DATASET_REGISTRY
        assert "video" in datasets.DATASET_REGISTRY
        assert "code" in datasets.DATASET_REGISTRY
        assert "text" in datasets.DATASET_REGISTRY
        assert len(datasets.DATASET_REGISTRY) == 4

    def test_each_is_callable(self):
        for name, fn in datasets.DATASET_REGISTRY.items():
            (s, t), (vs, vt) = fn(10)
            assert len(s) > 0
            assert len(t) > 0
