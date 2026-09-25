from pathlib import Path

from ultralytics import YOLO

from uav_xai.core.predictor import (
    ModelPredictor,
)


def test_yolo_detection_prediction():

    image_path = Path(
        "data/images/test.jpg"
    )

    if not image_path.exists():
        return

    model = YOLO(
        "yolo26n.pt"
    )

    predictor = ModelPredictor(
        model=model,
        task="detection",
    )

    result = predictor.predict(
        str(image_path)
    )

    assert result.task == (
        "detection"
    )

    assert result.detections is not None

    assert len(
        result.detections
    ) >= 1

    detection = (
        result.detections[0]
    )

    assert detection.class_id >= 0

    assert detection.class_name

    assert (
        0.0
        <= detection.confidence
        <= 1.0
    )

    assert len(
        detection.bbox
    ) == 4