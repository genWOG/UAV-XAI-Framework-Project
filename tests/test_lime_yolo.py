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
        self.batch_sizes = []

    def predict(self, source, conf, verbose):
        assert conf == 0.001
        assert verbose is False
        self.batch_sizes.append(len(source))

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
    assert explainer.target_score_mode == "confidence_iou"


def test_lime_yolo_detection_batches_vectorized_perturbations():
    model = FakeYoloModel([
        ([0, 0, 10, 10], 2, 0.72),
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
        num_segments=4,
        prediction_batch_size=4,
    )
    image = np.random.default_rng(7).integers(
        0, 256, size=(16, 16, 3), dtype=np.uint8
    )

    result = explainer.explain(model=model, input_data=image, target=target)

    assert result.attribution.shape == image.shape[:2]
    assert np.isfinite(result.attribution).all()
    assert result.metadata["prediction_batch_size"] == 4
    # One call scores the unperturbed image; the 10 LIME samples use 4/4/2.
    assert max(model.batch_sizes) <= 4
    assert sorted(model.batch_sizes).count(4) >= 2
    assert sum(model.batch_sizes) >= 11



def test_lime_yolo_uses_continuous_confidence_iou_score_below_threshold():
    model = FakeYoloModel([
        ([5, 0, 15, 10], 2, 0.72),
    ])
    target = XAITarget(
        task="detection",
        class_id=2,
        score=0.85,
        detection_index=0,
        bbox=[0, 0, 10, 10],
    )
    image = np.zeros((1, 32, 32, 3), dtype=np.uint8)

    continuous = LimeImageExplainer(
        preprocess=None,
        num_samples=10,
        target_iou_threshold=0.5,
        target_score_mode="confidence_iou",
    )._build_detection_predict_fn(model, target)(image)

    # Shifted box IoU = 1/3, so score remains informative below IoU 0.5.
    assert np.isclose(continuous[0, 1], 0.72 / 3.0, atol=1e-6)
    assert np.isclose(continuous[0].sum(), 1.0, atol=1e-6)

    thresholded = LimeImageExplainer(
        preprocess=None,
        num_samples=10,
        target_iou_threshold=0.5,
        target_score_mode="thresholded_confidence",
    )._build_detection_predict_fn(model, target)(image)
    assert thresholded[0, 1] == 0.0
