import time
from typing import Any, Callable

import numpy as np
import torch
from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox

from ..core.base import Explainer
from ..core.result import ExplanationResult


class LrpImageExplainer(Explainer):
    """
    LRP explainer for image classification using Zennit.

    Produces a spatial relevance map with shape [H, W].
    """

    def __init__(
        self,
        preprocess: Callable[[np.ndarray], torch.Tensor],
        epsilon: float = 1e-6,
        gamma: float = 0.1,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(
                "epsilon must be > 0."
            )

        self.preprocess = preprocess
        self.epsilon = epsilon
        self.gamma = gamma
        self.low = low
        self.high = high

    def explain(
        self,
        model: Any,
        input_data: Any,
        target: int | str | None = None,
    ) -> ExplanationResult:

        start_time = time.perf_counter()

        image = np.asarray(
            input_data
        )

        if image.ndim != 3:
            raise ValueError(
                "input_data must have shape [H, W, 3]."
            )

        if image.shape[2] != 3:
            raise ValueError(
                "Only RGB images are supported."
            )

        if not np.isfinite(
            image.astype(np.float32)
        ).all():
            raise ValueError(
                "Image contains NaN or Inf values."
            )

        if image.dtype != np.uint8:
            image = np.clip(
                image,
                0,
                255,
            ).astype(
                np.uint8
            )

        input_tensor = self.preprocess(
            image
        )

        if not isinstance(
            input_tensor,
            torch.Tensor,
        ):
            raise TypeError(
                "preprocess must return torch.Tensor."
            )

        if input_tensor.ndim != 3:
            raise ValueError(
                "preprocess must return tensor "
                "with shape [C, H, W]."
            )

        input_tensor = input_tensor.unsqueeze(
            0
        )

        try:
            device = next(
                model.parameters()
            ).device
            input_tensor = input_tensor.to(
                device
            )
        except StopIteration:
            pass

        model.eval()

        # Original prediction.
        with torch.no_grad():
            output = model(
                input_tensor
            )

        if output.ndim != 2:
            raise ValueError(
                "Model output must have shape "
                "[batch_size, num_classes]."
            )

        if output.shape[0] != 1:
            raise ValueError(
                "Only batch size 1 is supported."
            )

        if target is None:
            target = int(
                output.argmax(
                    dim=1
                ).item()
            )

        target = int(target)

        if not (
            0 <= target < output.shape[1]
        ):
            raise ValueError(
                f"Target class {target} is out of range."
            )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        confidence = float(
            probabilities[
                0,
                target,
            ].item()
        )

        # LRP composite.
        composite = EpsilonGammaBox(
            low=self.low,
            high=self.high,
            epsilon=self.epsilon,
            gamma=self.gamma,
        )

        # Select the target output neuron.
        target_vector = torch.zeros_like(
            output
        )

        target_vector[
            0,
            target
        ] = 1.0

        model.zero_grad(
            set_to_none=True
        )

        # Zennit replaces the standard backward rules
        # with the rules defined by the composite.
        with Gradient(
            model,
            composite,
        ) as attributor:

            _, relevance = attributor(
                input_tensor,
                target_vector,
            )

        if relevance is None:
            raise RuntimeError(
                "LRP relevance was not produced."
            )

        relevance = (
            relevance[
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # Collapse RGB/channel relevance into a spatial map.
        if relevance.ndim != 3:
            raise ValueError(
                "Expected relevance with shape [C, H, W]."
            )

        attribution = relevance.sum(
            axis=0
        ).astype(
            np.float32
        )

        runtime_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        return ExplanationResult(
            method="lrp",
            attribution=attribution,
            target=target,
            confidence=confidence,
            runtime_ms=runtime_ms,
            input_type="image",
            feature_names=None,
            metadata={
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "low": self.low,
                "high": self.high,
                "input_shape": list(
                    input_tensor.shape
                ),
                "attribution_shape": list(
                    attribution.shape
                ),
                "explainer": "lrp_image",
            },
        )