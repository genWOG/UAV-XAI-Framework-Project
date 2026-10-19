import time
from typing import Any, Callable

import numpy as np
import torch
from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox

from ..core.base import Explainer
from ..core.result import ExplanationResult
from ..core.target import XAITarget
from ..core.target_builder import TargetBuilder


class _DetectionTargetModule(torch.nn.Module):
    """Expose the differentiable YOLO target score as a scalar module output."""

    def __init__(
        self,
        model: Any,
        target: XAITarget,
        context: dict[str, Any],
    ) -> None:
        super().__init__()
        self.detector = TargetBuilder._get_detection_network(model)
        # Keep the original Ultralytics wrapper as a non-registered reference;
        # registering it separately would duplicate the same detector modules.
        object.__setattr__(self, "_model_ref", model)
        object.__setattr__(self, "_target_ref", target)
        object.__setattr__(self, "_context_ref", context)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        score = TargetBuilder.build(
            model=self._model_ref,
            input_data=input_tensor,
            target=self._target_ref,
            context=self._context_ref,
        )
        if not isinstance(score, torch.Tensor) or score.numel() != 1:
            raise ValueError("Detection target must be a single differentiable score.")
        return score.reshape(1)


class LrpImageExplainer(Explainer):
    """
    LRP explainer for image classification using Zennit.

    Produces a spatial relevance map with shape [H, W].
    """

    def __init__(
        self,
        preprocess: Callable[[np.ndarray], torch.Tensor] | None,
        epsilon: float = 1e-6,
        gamma: float = 0.1,
        low: float = 0.0,
        high: float = 1.0,
        context: dict[str, Any] | None = None,
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
        self.context = context or {}

    def _explain_detection(
        self,
        model: Any,
        target: XAITarget,
    ) -> ExplanationResult:
        start_time = time.perf_counter()
        if target.bbox is None:
            raise ValueError("Detection LRP requires target.bbox.")

        input_tensor = self.context.get("xai_input")
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Detection LRP requires context['xai_input'] as a torch.Tensor.")
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Detection LRP input must have shape [1, C, H, W].")

        detector = TargetBuilder._get_detection_network(model)
        detector.eval()
        try:
            device = next(detector.parameters()).device
        except StopIteration:
            device = input_tensor.device
        input_tensor = input_tensor.detach().to(device).clone().requires_grad_(True)

        target_model = _DetectionTargetModule(
            model=model,
            target=target,
            context=self.context,
        ).to(device)
        target_model.eval()
        composite = EpsilonGammaBox(
            low=self.low,
            high=self.high,
            epsilon=self.epsilon,
            gamma=self.gamma,
        )
        target_vector = torch.ones((1,), dtype=input_tensor.dtype, device=device)
        target_model.zero_grad(set_to_none=True)

        with Gradient(target_model, composite) as attributor:
            output, relevance = attributor(input_tensor, target_vector)

        if relevance is None:
            raise RuntimeError("LRP relevance was not produced for the detection target.")
        if relevance.ndim != 4 or relevance.shape[0] != 1:
            raise ValueError("Expected detection relevance with shape [1, C, H, W].")

        channel_relevance = relevance[0].detach().cpu().numpy()
        signed_attribution = channel_relevance.sum(axis=0).astype(np.float32)
        positive_relevance = np.maximum(channel_relevance, 0.0).sum(axis=0).astype(np.float32)
        negative_relevance = np.minimum(channel_relevance, 0.0).sum(axis=0).astype(np.float32)
        # Main map explains evidence supporting the target; signed totals remain
        # available in metadata so negative evidence is not silently forgotten.
        attribution = positive_relevance
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        target_score = float(output.detach().reshape(-1)[0].cpu().item())
        return ExplanationResult(
            method="lrp",
            attribution=attribution,
            target=target.class_id,
            confidence=float(target.score),
            runtime_ms=runtime_ms,
            input_type="image",
            feature_names=None,
            metadata={
                "epsilon": self.epsilon,
                "gamma": self.gamma,
                "low": self.low,
                "high": self.high,
                "input_shape": list(input_tensor.shape),
                "attribution_shape": list(attribution.shape),
                "explainer": "lrp_image",
                "target_task": "detection",
                "target_class_id": target.class_id,
                "detection_index": target.detection_index,
                "bbox": list(target.bbox),
                "target_score": target_score,
                "relevance_view": "positive_support",
                "signed_relevance_sum": float(signed_attribution.sum()),
                "positive_relevance_sum": float(positive_relevance.sum()),
                "negative_relevance_sum": float(negative_relevance.sum()),
            },
        )

    def explain(
        self,
        model: Any,
        input_data: Any,
        target: int | str | XAITarget | None = None,
        context: dict[str, Any] | None = None, 
    ) -> ExplanationResult:

        if context is not None:
            self.context = context

        if isinstance(target, XAITarget):
            if target.task != "detection":
                raise ValueError("LRP received an unsupported XAI target.")
            return self._explain_detection(model=model, target=target)

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