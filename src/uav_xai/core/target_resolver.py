from .prediction import PredictionResult
from .target import XAITarget


class TargetResolver:
    """
    Convert model prediction into a normalized XAI target.
    """

    @staticmethod
    def resolve(
        prediction: PredictionResult,
        target_type: str | None = None,
        detection_index: int = 0,
    ) -> XAITarget:

        task = (
            target_type
            or prediction.task
        )

        if task == "classification":
            return XAITarget(
                task="classification",
                class_id=prediction.class_id,
                score=prediction.confidence,
            )

        if task == "detection":

            detections = (
                prediction.detections
                or []
            )

            if not detections:
                raise ValueError(
                    "No detections available "
                    "for XAI target."
                )

            if not (
                0 <= detection_index
                < len(detections)
            ):
                raise IndexError(
                    f"Detection index "
                    f"{detection_index} is out of range."
                )

            detection = detections[
                detection_index
            ]

            return XAITarget(
                task="detection",
                class_id=detection.class_id,
                score=detection.confidence,
                detection_index=detection_index,
                bbox=list(
                    detection.bbox
                ),
            )

        raise ValueError(
            f"Unsupported target type: {task}"
        )