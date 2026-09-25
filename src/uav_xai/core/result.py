from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExplanationResult:
    """
    Unified result returned by any XAI method.

    Supports image, tabular and other input modalities.
    """

    method: str
    attribution: np.ndarray

    target: int | str | None = None
    confidence: float | None = None
    runtime_ms: float | None = None

    input_type: str = "unknown"

    feature_names: list[str] | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )