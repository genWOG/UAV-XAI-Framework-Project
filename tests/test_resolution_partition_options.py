from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uav_xai.core.target import XAITarget
from uav_xai.methods.lime_image import LimeImageExplainer
from uav_xai.methods.shap_image import ShapImageExplainer


def test_lime_working_image_resize_keeps_aspect_and_default_is_unchanged():
    image = np.zeros((1704, 2272, 3), dtype=np.uint8)
    default = LimeImageExplainer(preprocess=None, num_samples=2)
    assert default._resize_for_explanation(image) is image
    resized = LimeImageExplainer(
        preprocess=None, num_samples=2, explanation_image_size=960
    )._resize_for_explanation(image)
    assert resized.shape == (720, 960, 3)


def test_shap_working_image_resize_keeps_aspect_and_default_is_unchanged():
    image = np.zeros((1704, 2272, 3), dtype=np.uint8)
    default = ShapImageExplainer(preprocess=None, nsamples=2)
    assert default._resize_for_explanation(image) is image
    resized = ShapImageExplainer(
        preprocess=None, nsamples=2, explanation_image_size=640
    )._resize_for_explanation(image)
    assert resized.shape == (480, 640, 3)


def test_shap_algorithm_is_validated():
    ShapImageExplainer(preprocess=None, nsamples=2, algorithm="kernel")
    ShapImageExplainer(preprocess=None, nsamples=2, algorithm="partition")
    with pytest.raises(ValueError, match="algorithm"):
        ShapImageExplainer(preprocess=None, nsamples=2, algorithm="unknown")


def test_explanation_image_size_must_be_at_least_64():
    with pytest.raises(ValueError, match="explanation_image_size"):
        LimeImageExplainer(preprocess=None, explanation_image_size=32)
    with pytest.raises(ValueError, match="explanation_image_size"):
        ShapImageExplainer(preprocess=None, explanation_image_size=32)


class _FakeBoxes:
    def __init__(self, xyxy):
        self.xyxy = torch.tensor([xyxy], dtype=torch.float32)
        self.cls = torch.tensor([8.0], dtype=torch.float32)
        self.conf = torch.tensor([0.8], dtype=torch.float32)

    def __len__(self):
        return 1


class _FakeYolo:
    def predict(self, source, conf=0.001, verbose=False):
        images = source if isinstance(source, list) else [source]
        results = []
        for image in images:
            height, width = image.shape[:2]
            results.append(SimpleNamespace(
                boxes=_FakeBoxes([0.0, 0.0, float(width), float(height)]),
                names={8: "boat"},
            ))
        return results


def test_partition_shap_detection_smoke_and_original_resolution_output():
    image = np.full((64, 96, 3), 127, dtype=np.uint8)
    target = XAITarget(
        task="detection",
        class_id=8,
        score=0.8,
        detection_index=0,
        bbox=[8.0, 8.0, 88.0, 56.0],
    )
    explainer = ShapImageExplainer(
        preprocess=None,
        nsamples=16,
        num_segments=4,
        prediction_batch_size=4,
        explanation_image_size=64,
        algorithm="partition",
        baseline_mode="constant",
    )
    result = explainer.explain(_FakeYolo(), image, target)
    assert result.attribution.shape == image.shape[:2]
    assert result.metadata["algorithm"] == "partition"
    assert result.metadata["explanation_image_shape"] == [43, 64, 3]
    assert result.metadata["partition_max_evals"] >= 2 * result.metadata["num_segments"] + 1
    assert np.isfinite(result.attribution).all()
