from dataclasses import dataclass
from typing import Literal


TargetType = Literal[
    "classification",
    "detection",
]


@dataclass(frozen=True)
class XAITarget:
    """
    Normalized target for XAI.
    """

    task: TargetType

    class_id: int

    score: float

    detection_index: int | None = None

    bbox: list[float] | None = None