from typing import Any

import numpy as np
import torch

from .prediction import (
    DetectionResult,
    PredictionResult,
)


class ModelPredictor:
    """
    Unified interface for model inference.
    """

    def __init__(
        self,
        model: Any,
        task: str = "classification",
        confidence_threshold: float | None = None,
    ) -> None:
        self.model = model
        self.task = task.lower()
        self.confidence_threshold = confidence_threshold
        if confidence_threshold is not None and not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if confidence_threshold is not None:
            self.confidence_threshold = float(confidence_threshold)

        if hasattr(
            self.model,
            "eval",
        ):
            self.model.eval()

        if self.task not in {
            "classification",
            "detection",
        }:
            raise ValueError(
                f"Unsupported task: {task}. "
                "Expected classification or detection."
            )

    @torch.no_grad()
    def predict(
        self,
        input_data: Any,
    ) -> PredictionResult:
        """
        Run model inference.
        """

        if self.task == "detection":
            return self._predict_detection(
                input_data
            )

        return self._predict_classification(
            input_data
        )

    def _predict_classification(
        self,
        input_data: torch.Tensor,
    ) -> PredictionResult:

        if not isinstance(
            input_data,
            torch.Tensor,
        ):
            raise TypeError(
                "Classification input_data "
                "must be a torch.Tensor."
            )

        output = self.model(
            input_data
        )

        if output.ndim != 2:
            raise ValueError(
                "Expected model output with shape "
                "[batch_size, num_classes]."
            )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence, class_id = (
            probabilities.max(
                dim=1
            )
        )

        return PredictionResult(
            class_id=int(
                class_id[0].item()
            ),
            confidence=float(
                confidence[0].item()
            ),
            probabilities=(
                probabilities[0]
                .detach()
                .cpu()
                .numpy()
            ),
            task="classification",
        )

    def _predict_detection(
        self,
        input_data: Any,
    ) -> PredictionResult:
        """
        Run Ultralytics YOLO detection.
        """

        if not hasattr(
            self.model,
            "predict",
        ):
            raise TypeError(
                "Detection model must provide "
                "the Ultralytics predict() API."
            )

        predict_kwargs = {"source": input_data, "verbose": False}
        if self.confidence_threshold is not None:
            predict_kwargs["conf"] = self.confidence_threshold
        results = self.model.predict(**predict_kwargs)

        if not results:
            raise RuntimeError(
                "Detection model returned no results."
            )

        result = results[0]

        boxes = result.boxes

        detections: list[
            DetectionResult
        ] = []

        if boxes is not None:

            xyxy = boxes.xyxy

            confidences = boxes.conf

            class_ids = boxes.cls

            for index in range(
                len(boxes)
            ):

                class_id = int(
                    class_ids[
                        index
                    ].item()
                )

                confidence = float(
                    confidences[
                        index
                    ].item()
                )

                bbox = [
                    float(value)
                    for value in (
                        xyxy[
                            index
                        ]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                ]

                class_name = result.names[
                    class_id
                ]

                detections.append(
                    DetectionResult(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                    )
                )

        if not detections:
            return PredictionResult(
                class_id=-1,
                confidence=0.0,
                probabilities=None,
                task="detection",
                detections=[],
            )

        # The first YOLO detection is used as the
        # primary prediction for backward-compatible
        # confidence/class fields.
        primary = detections[0]

        return PredictionResult(
            class_id=primary.class_id,
            confidence=primary.confidence,
            probabilities=None,
            task="detection",
            detections=detections,
        )