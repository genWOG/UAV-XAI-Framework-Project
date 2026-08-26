from typing import Any

import numpy as np
import torch

from .prediction import PredictionResult


class ModelPredictor:
    """
    Unified interface for model inference.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        input_data: torch.Tensor,
    ) -> PredictionResult:
        """
        Run model inference.

        Parameters
        ----------
        input_data:
            Tensor with shape [batch, channels, height, width].

        Returns
        -------
        PredictionResult
            Predicted class, confidence and all probabilities.
        """

        output = self.model(input_data)

        if output.ndim != 2:
            raise ValueError(
                "Expected model output with shape "
                "[batch_size, num_classes]."
            )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence, class_id = probabilities.max(
            dim=1
        )

        return PredictionResult(
            class_id=int(class_id[0].item()),
            confidence=float(confidence[0].item()),
            probabilities=probabilities[0]
            .detach()
            .cpu()
            .numpy(),
        )