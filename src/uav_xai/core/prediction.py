from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    """
    Single object detection.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


@dataclass
class PredictionResult:
    """
    Standardized result of model inference.

    For classification:
        class_id
        confidence
        probabilities

    For detection:
        detections
    """

    class_id: int
    confidence: float
    probabilities: np.ndarray | None = None

    task: str = "classification"

    detections: list[DetectionResult] | None = None