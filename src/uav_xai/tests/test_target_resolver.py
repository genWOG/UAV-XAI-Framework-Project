from uav_xai.core.prediction import (
    DetectionResult,
    PredictionResult,
)
from uav_xai.core.target_resolver import (
    TargetResolver,
)


def test_resolves_classification_target():

    prediction = PredictionResult(
        class_id=751,
        confidence=0.6284,
        probabilities=None,
        task="classification",
    )

    target = TargetResolver.resolve(
        prediction
    )

    assert target.task == "classification"
    assert target.class_id == 751
    assert target.score == 0.6284
    assert target.detection_index is None
    assert target.bbox is None


def test_resolves_detection_target():

    prediction = PredictionResult(
        class_id=2,
        confidence=0.9602,
        probabilities=None,
        task="detection",
        detections=[
            DetectionResult(
                class_id=2,
                class_name="car",
                confidence=0.9602,
                bbox=[
                    162.4,
                    120.9,
                    662.7,
                    366.2,
                ],
            ),
        ],
    )

    target = TargetResolver.resolve(
        prediction,
        target_type="detection",
        detection_index=0,
    )

    assert target.task == "detection"
    assert target.class_id == 2
    assert target.score == 0.9602
    assert target.detection_index == 0
    assert target.bbox == [
        162.4,
        120.9,
        662.7,
        366.2,
    ]