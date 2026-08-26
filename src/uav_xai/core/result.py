from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExplanationResult:
    """
    Standardized result returned by every XAI method.

    Attributes
    ----------
    method:
        Name of the explanation method, e.g. "grad_cam".
    attribution:
        2D or 3D attribution map produced by the explainer.
    target:
        Target class/action for which the explanation was generated.
    confidence:
        Model confidence for the target.
    runtime_ms:
        Explanation generation time in milliseconds.
    metadata:
        Additional method-specific information.
    """

    method: str
    attribution: np.ndarray

    target: int | str | None = None
    confidence: float | None = None
    runtime_ms: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)