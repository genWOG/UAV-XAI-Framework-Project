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

    for metric_result in (deletion, insertion):
        assert len(metric_result.class_confidences) == 5
        assert len(metric_result.best_iou_candidate_confidences) == 5
        assert len(metric_result.has_target_class_detection) == 5
        assert len(metric_result.target_detection_matched) == 5
        assert set(metric_result.to_dict()) == {
            "fractions",
            "detection_scores",
            "localization_scores",
            "localization_auc",
            "class_confidences",
            "best_iou_candidate_confidences",
            "ious",
            "has_target_class_detection",
            "target_detection_matched",
        }

    assert deletion.confidences[0] > deletion.confidences[-1]
    assert insertion.confidences[0] < insertion.confidences[-1]

    assert np.isfinite(deletion.auc)
    assert np.isfinite(insertion.auc)

def test_match_target_detection_rejects_low_iou():
    prediction = PredictionResult(
        class_id=2,
        confidence=0.95,
        task="detection",
        detections=[
            DetectionResult(
                class_id=2,
                class_name="car",
                confidence=0.95,
                bbox=[9.0, 9.0, 19.0, 19.0],
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

    assert 0.0 < iou < 0.5
    assert confidence == 0.0

def test_match_target_detection_selects_target_by_iou():
    prediction = PredictionResult(
        class_id=2,
        confidence=0.99,
        task="detection",
        detections=[
            DetectionResult(
                class_id=2,
                class_name="car",
                confidence=0.99,
                bbox=[50.0, 50.0, 60.0, 60.0],
            ),
            DetectionResult(
                class_id=2,
                class_name="car",
                confidence=0.72,
                bbox=[0.0, 0.0, 10.0, 10.0],
            ),
        ],
    )

    target = XAITarget(
        task="detection",
        class_id=2,
        score=0.85,
        bbox=[0.0, 0.0, 10.0, 10.0],
    )

    confidence, iou = match_target_detection(
        prediction,
        target,
    )

    assert confidence == 0.72
    assert iou == 1.0

class SequencePredictor:
    """Return predefined predictions in order, independently of the input."""

    def __init__(self, predictions):
        self.predictions = iter(predictions)

    def predict(self, input_data):
        return next(self.predictions)


def _car(confidence, bbox):
    return DetectionResult(
        class_id=2,
        class_name="car",
        confidence=confidence,
        bbox=bbox,
    )


def _prediction_with_detections(detections):
    return PredictionResult(
        class_id=2,
        confidence=max((d.confidence for d in detections), default=0.0),
        task="detection",
        detections=detections,
    )


def _run_deletion_sequence(predictions, iou_threshold=0.5):
    return compute_detection_deletion(
        predictor=SequencePredictor(predictions),
        input_tensor=torch.ones(1, 3, 2, 2),
        attribution=np.ones((2, 2), dtype=np.float32),
        target=XAITarget(
            task="detection",
            class_id=2,
            score=0.9,
            bbox=[0.0, 0.0, 10.0, 10.0],
        ),
        steps=1,
        baseline_value=0.0,
        iou_threshold=iou_threshold,
    )


def test_detection_diagnostics_confidence_drops_bbox_stays():
    result = _run_deletion_sequence([
        _prediction_with_detections([
            _car(0.9, [0.0, 0.0, 10.0, 10.0])
        ]),
        _prediction_with_detections([
            _car(0.3, [0.0, 0.0, 10.0, 10.0])
        ]),
    ])

    np.testing.assert_allclose(result.class_confidences, [0.9, 0.3])
    np.testing.assert_allclose(result.best_iou_candidate_confidences, [0.9, 0.3])
    np.testing.assert_allclose(result.ious, [1.0, 1.0])
    np.testing.assert_allclose(result.detection_scores, [0.9, 0.3])
    assert result.target_detection_matched.tolist() == [True, True]


def test_detection_diagnostics_bbox_shifts_at_constant_confidence():
    result = _run_deletion_sequence([
        _prediction_with_detections([
            _car(0.8, [0.0, 0.0, 10.0, 10.0])
        ]),
        _prediction_with_detections([
            _car(0.8, [5.0, 0.0, 15.0, 10.0])
        ]),
    ])

    np.testing.assert_allclose(result.class_confidences, [0.8, 0.8])
    np.testing.assert_allclose(result.ious, [1.0, 1.0 / 3.0])
    np.testing.assert_allclose(result.detection_scores, [0.8, 0.0])
    assert result.target_detection_matched.tolist() == [True, False]
    assert result.has_target_class_detection.tolist() == [True, True]


def test_detection_diagnostics_detection_disappears():
    result = _run_deletion_sequence([
        _prediction_with_detections([
            _car(0.9, [0.0, 0.0, 10.0, 10.0])
        ]),
        _prediction_with_detections([]),
    ])

    assert result.has_target_class_detection.tolist() == [True, False]
    assert result.target_detection_matched.tolist() == [True, False]
    np.testing.assert_allclose(result.class_confidences, [0.9, 0.0])
    np.testing.assert_allclose(result.detection_scores, [0.9, 0.0])


def test_detection_diagnostics_other_same_class_object_remains():
    result = _run_deletion_sequence([
        _prediction_with_detections([
            _car(0.9, [0.0, 0.0, 10.0, 10.0])
        ]),
        _prediction_with_detections([
            _car(0.95, [30.0, 30.0, 40.0, 40.0])
        ]),
    ])

    assert result.has_target_class_detection.tolist() == [True, True]
    assert result.target_detection_matched.tolist() == [True, False]
    assert result.class_confidences[1] == 0.95
    assert result.best_iou_candidate_confidences[1] == 0.95
    assert result.ious[1] == 0.0
    assert result.detection_scores[1] == 0.0


def test_detection_deletion_uses_configured_iou_threshold():
    result = _run_deletion_sequence([
        _prediction_with_detections([
            _car(0.8, [0.0, 0.0, 10.0, 10.0])
        ]),
        _prediction_with_detections([
            _car(0.8, [5.0, 0.0, 15.0, 10.0])
        ]),
    ], iou_threshold=0.3)

    assert result.target_detection_matched.tolist() == [True, True]
    np.testing.assert_allclose(result.detection_scores, [0.8, 0.8])

def test_zero_iou_is_not_reported_as_matched_with_zero_threshold():
    result = _run_deletion_sequence([
        _prediction_with_detections([
            _car(0.9, [0.0, 0.0, 10.0, 10.0])
        ]),
        _prediction_with_detections([
            _car(0.95, [30.0, 30.0, 40.0, 40.0])
        ]),
    ], iou_threshold=0.0)

    assert result.has_target_class_detection.tolist() == [True, True]
    assert result.target_detection_matched.tolist() == [True, False]
    np.testing.assert_allclose(result.ious, [1.0, 0.0])
    # Keep legacy score semantics even at the boundary threshold.
    np.testing.assert_allclose(result.detection_scores, [0.9, 0.0])



def test_localization_weighted_curve_retains_subthreshold_iou_signal():
    result = _run_deletion_sequence([
        _prediction_with_detections([
            _car(0.8, [0.0, 0.0, 10.0, 10.0])
        ]),
        _prediction_with_detections([
            _car(0.8, [5.0, 0.0, 15.0, 10.0])
        ]),
    ])

    # Existing metric stays thresholded for backward compatibility.
    np.testing.assert_allclose(result.detection_scores, [0.8, 0.0])
    # The additional curve remains continuous as localization degrades.
    np.testing.assert_allclose(result.localization_scores, [0.8, 0.8 / 3.0])
    assert np.isclose(result.localization_auc, (0.8 + 0.8 / 3.0) / 2.0)
    payload = result.to_dict()
    np.testing.assert_allclose(payload["localization_scores"], [0.8, 0.8 / 3.0])
    assert np.isclose(payload["localization_auc"], result.localization_auc)
