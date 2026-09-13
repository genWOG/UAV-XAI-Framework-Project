import numpy as np
import torch

from uav_xai.core.prediction import (
    DetectionResult,
    PredictionResult,
)
from uav_xai.core.target import XAITarget
from uav_xai.metrics.detection import (
    compute_detection_deletion,
    compute_detection_insertion,
    match_target_detection,
)


class FakePredictor:
    def predict(self, input_data):
        value = float(
            input_data.mean().item()
        )

        confidence = max(
            0.0,
            min(1.0, value),
        )

        return PredictionResult(
            class_id=2,
            confidence=confidence,
            task="detection",
            detections=[
                DetectionResult(
                    class_id=2,
                    class_name="car",
                    confidence=confidence,
                    bbox=[0.0, 0.0, 10.0, 10.0],
                )
            ],
        )


def test_match_target_detection():
    prediction = PredictionResult(
        class_id=2,
        confidence=0.9,
        task="detection",
        detections=[
            DetectionResult(
                class_id=2,
                class_name="car",
                confidence=0.75,
                bbox=[0.0, 0.0, 10.0, 10.0],
            ),
            DetectionResult(
                class_id=1,
                class_name="person",
                confidence=0.99,
                bbox=[0.0, 0.0, 10.0, 10.0],
            ),
        ],
    )

    target = XAITarget(
        task="detection",
        class_id=2,
        score=0.9,
        bbox=[0.0, 0.0, 10.0, 10.0],
    )

    confidence, iou = match_target_detection(
        prediction,
        target,
    )

    assert confidence == 0.75
    assert iou == 1.0


def test_detection_deletion_and_insertion():
    predictor = FakePredictor()

    input_tensor = torch.ones(
        1,
        3,
        8,
        8,
    )

    attribution = np.ones(
        (8, 8),
        dtype=np.float32,
    )

    target = XAITarget(
        task="detection",
        class_id=2,
        score=1.0,
        bbox=[0.0, 0.0, 10.0, 10.0],
    )

    deletion = compute_detection_deletion(
        predictor=predictor,
        input_tensor=input_tensor,
        attribution=attribution,
        target=target,
        steps=4,
        baseline_value=0.0,
    )

    insertion = compute_detection_insertion(
        predictor=predictor,
        input_tensor=input_tensor,
        attribution=attribution,
        target=target,
        steps=4,
        baseline_value=0.0,
    )

    assert len(deletion.fractions) == 5
    assert len(deletion.confidences) == 5
    assert len(deletion.ious) == 5

    assert len(insertion.fractions) == 5
    assert len(insertion.confidences) == 5
    assert len(insertion.ious) == 5

    assert deletion.confidences[0] > deletion.confidences[-1]
    assert insertion.confidences[0] < insertion.confidences[-1]

    assert np.isfinite(deletion.auc)
    assert np.isfinite(insertion.auc)