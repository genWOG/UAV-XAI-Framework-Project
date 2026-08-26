from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResult:
    """
    Standardized result of model inference.
    """

    class_id: int
    confidence: float
    probabilities: np.ndarray