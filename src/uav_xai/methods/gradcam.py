import time
from typing import Any

import numpy as np
import torch

from ..core.base import Explainer
from ..core.result import ExplanationResult


class GradCAMExplainer(Explainer):
    """
    Grad-CAM implementation for convolutional neural networks.

    The method uses activations and gradients from a selected
    convolutional layer to produce a spatial importance map.
    """

    def __init__(self, target_layer: torch.nn.Module) -> None:
        self.target_layer = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )

        self._backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._activations = output.detach()

    def _backward_hook(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_output[0] is not None:
            self._gradients = grad_output[0].detach()

    def explain(
        self,
        model: Any,
        input_data: torch.Tensor,
        target: int | None = None,
    ) -> ExplanationResult:

        if not isinstance(input_data, torch.Tensor):
            raise TypeError(
                "input_data must be a torch.Tensor"
            )

        model.eval()

        device = next(model.parameters()).device
        input_data = input_data.to(device)

        start_time = time.perf_counter()

        model.zero_grad(set_to_none=True)

        output = model(input_data)

        if output.ndim != 2:
            raise ValueError(
                "Expected model output with shape "
                "[batch_size, num_classes]."
            )

        if target is None:
            target = int(output.argmax(dim=1).item())

        target_score = output[:, target].sum()
        target_score.backward()

        if self._activations is None:
            raise RuntimeError(
                "Forward activations were not captured."
            )

        if self._gradients is None:
            raise RuntimeError(
                "Backward gradients were not captured."
            )

        # Global average pooling of gradients.
        weights = self._gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted sum of feature maps.
        cam = (
            weights * self._activations
        ).sum(dim=1)

        # ReLU: only positive influence is retained.
        cam = torch.relu(cam)

        # Normalize each heatmap to [0, 1].
        cam_min = cam.amin(dim=(1, 2), keepdim=True)
        cam_max = cam.amax(dim=(1, 2), keepdim=True)

        cam = (cam - cam_min) / (
            cam_max - cam_min + 1e-8
        )

        heatmap = cam[0].cpu().numpy()

        probabilities = torch.softmax(output, dim=1)
        confidence = float(
            probabilities[0, target].item()
        )

        runtime_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        return ExplanationResult(
            method="grad_cam",
            attribution=heatmap,
            target=target,
            confidence=confidence,
            runtime_ms=runtime_ms,
            input_type="image",
            metadata={
                "input_shape": tuple(input_data.shape),
                "output_shape": tuple(output.shape),
                "target_score": float(
                    target_score.detach().item()
                ),
            },
        )

    def close(self) -> None:
        """
        Remove registered PyTorch hooks.
        """
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass