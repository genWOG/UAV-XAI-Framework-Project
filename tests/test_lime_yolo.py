import numpy as np
import torch

from types import SimpleNamespace

from uav_xai.core.target import XAITarget
from uav_xai.methods.lime_image import LimeImageExplainer


class FakeBoxes:
    def __init__(self, detections):
        self.xyxy = torch.tensor(
            [d[0] for d in detections],
            dtype=torch.float32,
        )
        self.cls = torch.tensor(
            [d[1] for d in detections],
            dtype=torch.float32,
        )
        self.conf = torch.tensor(
            [d[2] for d in detections],
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.cls)


class FakeYoloModel:
    def __init__(self, detections):
        self.detections = detections

    def predict(self, source, conf, verbose):
        assert conf == 0.001
        assert verbose is False

        return [
            SimpleNamespace(
                boxes=FakeBoxes(self.detections)
            )
            for _ in source
        ]


def test_lime_yolo_uses_target_bbox_not_highest_confidence():
    model = FakeYoloModel([
        ([50, 50, 60, 60], 2, 0.99),
        ([0, 0, 10, 10], 2, 0.72),
    ])

    target = XAITarget(
        task="detection",
        class_id=2,
        score=0.85,
        detection_index=1,
        bbox=[0, 0, 10, 10],
    )

    explainer = LimeImageExplainer(
        preprocess=None,
        num_samples=10,
        target_iou_threshold=0.5,
    )

    predict_fn = explainer._build_detection_predict_fn(
        model,
        target,
    )

    images = np.zeros(
        (2, 32, 32, 3),
        dtype=np.uint8,
    )

    scores = predict_fn(images)

    assert scores.shape == (2, 2)
    assert np.allclose(
        scores[:, 1],
        [0.72, 0.72],
    )


def test_lime_yolo_returns_zero_when_target_is_not_matched():
    model = FakeYoloModel([
        ([50, 50, 60, 60], 2, 0.99),
    ])

    target = XAITarget(
        task="detection",
        class_id=2,
        score=0.85,
        detection_index=0,
        bbox=[0, 0, 10, 10],
    )

    explainer = LimeImageExplainer(
        preprocess=None,
        num_samples=10,
        target_iou_threshold=0.5,
    )

    predict_fn = explainer._build_detection_predict_fn(
        model,
        target,
    )

    scores = predict_fn(
        np.zeros(
            (1, 32, 32, 3),
            dtype=np.uint8,
        )
    )

    assert scores[0, 1] == 0.0
    assert scores[0, 0] == 1.0