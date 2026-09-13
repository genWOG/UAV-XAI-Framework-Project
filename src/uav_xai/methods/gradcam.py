import time
from typing import Any

import torch
import torch.nn.functional as F

from ..core.base import Explainer
from ..core.result import ExplanationResult
from ..core.target import XAITarget
from ..core.target_builder import TargetBuilder


class GradCAMExplainer(Explainer):
    """
    Grad-CAM / LayerCAM-style explainer for classification
    and object detection.

    Classification
    --------------
    Standard Grad-CAM:

        ReLU(sum_k(mean spatial gradient_k * activation_k))

    Detection
    ---------
    Spatial LayerCAM-style relevance:

        ReLU(sum_k(ReLU(gradient_k) * activation_k))

    For YOLO-like detectors multiple feature scales can be used,
    for example:

        P3 -> layer 16
        P4 -> layer 19
        P5 -> layer 22

    Detection CAMs are:
        1. calculated independently for every feature scale;
        2. resized to input resolution;
        3. softly denoised;
        4. normalized independently;
        5. combined with scale weights;
        6. optionally biased toward the selected detection bbox;
        7. normalized again.
    """

    def __init__(
        self,
        target_layer: (
            torch.nn.Module
            | list[torch.nn.Module]
            | tuple[torch.nn.Module, ...]
        ),
    ) -> None:

        if isinstance(
            target_layer,
            (list, tuple),
        ):
            self.target_layers = list(
                target_layer
            )
        else:
            self.target_layers = [
                target_layer
            ]

        if not self.target_layers:
            raise ValueError(
                "At least one Grad-CAM target layer is required."
            )

        for layer in self.target_layers:
            if not isinstance(
                layer,
                torch.nn.Module,
            ):
                raise TypeError(
                    "Every Grad-CAM target layer "
                    "must be a torch.nn.Module."
                )

        self._activations: dict[
            int,
            torch.Tensor,
        ] = {}

        self._gradients: dict[
            int,
            torch.Tensor,
        ] = {}

        self._forward_handles: list[Any] = []
        self._backward_handles: list[Any] = []

        for index, layer in enumerate(
            self.target_layers
        ):
            self._forward_handles.append(
                layer.register_forward_hook(
                    self._make_forward_hook(index)
                )
            )

            self._backward_handles.append(
                layer.register_full_backward_hook(
                    self._make_backward_hook(index)
                )
            )

    # ============================================================
    # Hooks
    # ============================================================

    def _make_forward_hook(
        self,
        index: int,
    ):
        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:

            if not isinstance(
                output,
                torch.Tensor,
            ):
                raise TypeError(
                    "Grad-CAM target layer must "
                    "produce a torch.Tensor."
                )

            self._activations[index] = output

        return hook

    def _make_backward_hook(
        self,
        index: int,
    ):
        def hook(
            module: torch.nn.Module,
            grad_input: tuple[
                torch.Tensor | None,
                ...,
            ],
            grad_output: tuple[
                torch.Tensor | None,
                ...,
            ],
        ) -> None:

            if not grad_output:
                return

            gradient = grad_output[0]

            if gradient is not None:
                self._gradients[index] = gradient

        return hook

    # ============================================================
    # Model
    # ============================================================

    @staticmethod
    def _resolve_gradient_model(
        model: Any,
    ) -> torch.nn.Module:
        """
        Resolve the actual PyTorch model.

        Ultralytics:
            YOLO -> YOLO.model -> DetectionModel
        """

        wrapper_model = getattr(
            model,
            "model",
            None,
        )

        if (
            isinstance(
                wrapper_model,
                torch.nn.Module,
            )
            and model.__class__.__name__
            != "DetectionModel"
        ):
            gradient_model = wrapper_model
        else:
            gradient_model = model

        if not isinstance(
            gradient_model,
            torch.nn.Module,
        ):
            raise TypeError(
                "Grad-CAM requires a torch.nn.Module model."
            )

        return gradient_model

    # ============================================================
    # Normalization
    # ============================================================

    @staticmethod
    def _normalize_cam(
        cam: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize [B, H, W] CAM tensor to [0, 1].
        """

        cam_min = cam.amin(
            dim=(1, 2),
            keepdim=True,
        )

        cam_max = cam.amax(
            dim=(1, 2),
            keepdim=True,
        )

        denominator = (
            cam_max
            - cam_min
            + 1e-8
        )

        cam = (
            cam - cam_min
        ) / denominator

        return torch.clamp(
            cam,
            0.0,
            1.0,
        )

    # ============================================================
    # Smooth CAM
    # ============================================================

    @staticmethod
    def _smooth_cam(
        cam: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mild spatial denoising.

        Uses two small average-pooling passes rather than a
        hard threshold, preserving broader object structure.
        """

        cam = F.avg_pool2d(
            cam.unsqueeze(1),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(1)

        cam = F.avg_pool2d(
            cam.unsqueeze(1),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(1)

        return cam

    # ============================================================
    # Layer CAM
    # ============================================================

    def _compute_layer_cam(
        self,
        index: int,
        output_height: int,
        output_width: int,
        spatial_mode: bool,
    ) -> torch.Tensor:

        activations = self._activations.get(
            index
        )

        gradients = self._gradients.get(
            index
        )

        if activations is None:
            raise RuntimeError(
                f"Forward activations were not captured "
                f"for target layer {index}."
            )

        if gradients is None:
            raise RuntimeError(
                f"Backward gradients were not captured "
                f"for target layer {index}."
            )

        if activations.ndim != 4:
            raise ValueError(
                "Grad-CAM activations must have shape "
                "[batch, channels, height, width]. "
                f"Got {tuple(activations.shape)}."
            )

        if gradients.ndim != 4:
            raise ValueError(
                "Grad-CAM gradients must have shape "
                "[batch, channels, height, width]. "
                f"Got {tuple(gradients.shape)}."
            )

        if activations.shape != gradients.shape:
            raise ValueError(
                "Grad-CAM activations and gradients "
                "must have identical shapes. "
                f"Activations="
                f"{tuple(activations.shape)}, "
                f"gradients="
                f"{tuple(gradients.shape)}."
            )

        # --------------------------------------------------------
        # Classification: classic Grad-CAM.
        # --------------------------------------------------------

        if not spatial_mode:

            channel_weights = gradients.mean(
                dim=(2, 3),
                keepdim=True,
            )

            cam = (
                channel_weights
                * activations
            ).sum(
                dim=1
            )

            cam = torch.relu(
                cam
            )

        # --------------------------------------------------------
        # Detection: LayerCAM-style spatial relevance.
        # --------------------------------------------------------

        else:

            positive_gradients = torch.relu(
                gradients
            )

            positive_activations = torch.relu(
                activations
            )

            cam = (
                positive_gradients
                * positive_activations
            ).sum(
                dim=1
            )

            cam = torch.relu(
                cam
            )

        # --------------------------------------------------------
        # Spatial smoothing before interpolation.
        # --------------------------------------------------------

        cam = self._smooth_cam(
            cam
        )

        # --------------------------------------------------------
        # Resize to model input resolution.
        # --------------------------------------------------------

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(
                output_height,
                output_width,
            ),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # --------------------------------------------------------
        # Normalize independently for each scale.
        # --------------------------------------------------------

        cam = self._normalize_cam(
            cam
        )

        return cam

    # ============================================================
    # Detection bbox prior
    # ============================================================

    @staticmethod
    def _apply_soft_bbox_prior(
        cam: torch.Tensor,
        target: XAITarget,
        context: dict[str, Any] | None,
    ) -> torch.Tensor:
        """
        Blend model attribution with a soft object prior.

        The bbox is used only as a weak spatial prior.
        Model attribution remains dominant.
        """

        if (
            target.bbox is None
            or context is None
        ):
            return cam

        letterbox = context.get(
            "letterbox"
        )

        if letterbox is None:
            return cam

        h = int(
            letterbox["input_height"]
        )

        w = int(
            letterbox["input_width"]
        )

        scale = float(
            letterbox["scale"]
        )

        pad_x = float(
            letterbox["pad_x"]
        )

        pad_y = float(
            letterbox["pad_y"]
        )

        x1, y1, x2, y2 = (
            float(v)
            for v in target.bbox
        )

        x1 = x1 * scale + pad_x
        y1 = y1 * scale + pad_y
        x2 = x2 * scale + pad_x
        y2 = y2 * scale + pad_y

        yy = torch.arange(
            h,
            device=cam.device,
            dtype=cam.dtype,
        )

        xx = torch.arange(
            w,
            device=cam.device,
            dtype=cam.dtype,
        )

        grid_y, grid_x = torch.meshgrid(
            yy,
            xx,
            indexing="ij",
        )

        inside = (
            (grid_x >= x1)
            & (grid_x <= x2)
            & (grid_y >= y1)
            & (grid_y <= y2)
        ).to(
            cam.dtype
        )

        # Distance from bbox center.
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        bw = max(
            x2 - x1,
            1.0,
        )

        bh = max(
            y2 - y1,
            1.0,
        )

        dx = (
            (grid_x - cx)
            / (bw * 0.5)
        )

        dy = (
            (grid_y - cy)
            / (bh * 0.5)
        )

        radial = torch.sqrt(
            dx.pow(2)
            + dy.pow(2)
        )

        radial = torch.clamp(
            1.0 - radial,
            0.0,
            1.0,
        )

        # Weak object prior:
        # inside bbox -> 0.30..0.50
        # outside bbox -> smoothly decays.
        prior = (
            0.15
            + 0.35 * inside
            + 0.20 * radial
        )

        prior = prior.unsqueeze(
            0
        )

        # Keep model attribution dominant.
        return (
            cam * 0.75
            + prior * 0.25
        )

    # ============================================================
    # Explain
    # ============================================================

    def explain(
        self,
        model: Any,
        input_data: torch.Tensor,
        target: int | XAITarget | None = None,
        context: dict[str, Any] | None = None,
    ) -> ExplanationResult:
        """
        Generate a Grad-CAM explanation.
        """

        if not isinstance(
            input_data,
            torch.Tensor,
        ):
            raise TypeError(
                "input_data must be a torch.Tensor."
            )

        if input_data.ndim != 4:
            raise ValueError(
                "input_data must have shape "
                "[batch, channels, height, width]."
            )

        gradient_model = (
            self._resolve_gradient_model(
                model
            )
        )

        gradient_model.eval()

        try:
            device = next(
                gradient_model.parameters()
            ).device
        except StopIteration as exc:
            raise ValueError(
                "Model must contain at least one parameter."
            ) from exc

        input_data = input_data.to(
            device
        )

        start_time = time.perf_counter()

        # Cached explainers need clean state.
        self._activations.clear()
        self._gradients.clear()

        gradient_model.zero_grad(
            set_to_none=True
        )

        xai_target: (
            XAITarget | None
        ) = None

        output = None

        # --------------------------------------------------------
        # Detection / structured target
        # --------------------------------------------------------

        if isinstance(
            target,
            XAITarget,
        ):

            xai_target = target

            target_score = (
                TargetBuilder.build(
                    model=model,
                    input_data=input_data,
                    target=xai_target,
                    context=context,
                )
            )

            if not isinstance(
                target_score,
                torch.Tensor,
            ):
                raise TypeError(
                    "TargetBuilder must return "
                    "a torch.Tensor."
                )

            if target_score.ndim != 0:
                target_score = (
                    target_score.sum()
                )

            if not target_score.requires_grad:
                raise RuntimeError(
                    "XAI target is not differentiable."
                )

            target_class_id = (
                xai_target.class_id
            )

            confidence = float(
                xai_target.score
            )

            spatial_mode = True

        # --------------------------------------------------------
        # Classification target
        # --------------------------------------------------------

        else:

            output = gradient_model(
                input_data
            )

            if not isinstance(
                output,
                torch.Tensor,
            ):
                raise TypeError(
                    "Classification model must return "
                    "a torch.Tensor."
                )

            if output.ndim != 2:
                raise ValueError(
                    "Expected model output with shape "
                    "[batch_size, num_classes]."
                )

            if target is None:
                target_class_id = int(
                    output.argmax(
                        dim=1
                    ).item()
                )
            else:
                target_class_id = int(
                    target
                )

            if not (
                0 <= target_class_id
                < output.shape[1]
            ):
                raise ValueError(
                    f"Target class "
                    f"{target_class_id} is out of range."
                )

            target_score = output[
                :,
                target_class_id,
            ].sum()

            confidence = float(
                torch.softmax(
                    output,
                    dim=1,
                )[
                    0,
                    target_class_id,
                ].item()
            )

            spatial_mode = False

        # --------------------------------------------------------
        # Backward
        # --------------------------------------------------------

        if not target_score.requires_grad:
            raise RuntimeError(
                "Grad-CAM target score does not "
                "require gradients."
            )

        target_score.backward()

        if not self._activations:
            raise RuntimeError(
                "No Grad-CAM activations were captured."
            )

        if not self._gradients:
            raise RuntimeError(
                "No Grad-CAM gradients were captured."
            )

        output_height = int(
            input_data.shape[2]
        )

        output_width = int(
            input_data.shape[3]
        )

        # --------------------------------------------------------
        # Compute all feature-scale CAMs.
        # --------------------------------------------------------

        layer_cams: list[
            torch.Tensor
        ] = []

        for index in range(
            len(self.target_layers)
        ):

            cam = (
                self._compute_layer_cam(
                    index=index,
                    output_height=output_height,
                    output_width=output_width,
                    spatial_mode=spatial_mode,
                )
            )

            layer_cams.append(
                cam
            )

        # --------------------------------------------------------
        # Combine scales.
        # --------------------------------------------------------

        if len(layer_cams) == 1:

            combined_cam = (
                layer_cams[0]
            )

        else:

            if spatial_mode:

                # P3 / P4 / P5.
                scale_weights = [
                    0.50,
                    0.30,
                    0.20,
                ]

            else:

                scale_weights = [
                    1.0,
                    1.0,
                    1.0,
                ]

            if len(layer_cams) <= len(
                scale_weights
            ):

                weights = torch.tensor(
                    scale_weights[
                        :len(layer_cams)
                    ],
                    dtype=layer_cams[
                        0
                    ].dtype,
                    device=layer_cams[
                        0
                    ].device,
                )

            else:

                weights = torch.ones(
                    len(layer_cams),
                    dtype=layer_cams[
                        0
                    ].dtype,
                    device=layer_cams[
                        0
                    ].device,
                )

            weights /= weights.sum()

            combined_cam = torch.zeros_like(
                layer_cams[0]
            )

            for (
                cam,
                weight,
            ) in zip(
                layer_cams,
                weights,
            ):
                combined_cam += (
                    cam * weight
                )

            combined_cam = (
                self._normalize_cam(
                    combined_cam
                )
            )

        # --------------------------------------------------------
        # Detection-specific spatial stabilization.
        # --------------------------------------------------------

        if (
            xai_target is not None
            and spatial_mode
        ):

            combined_cam = (
                self._apply_soft_bbox_prior(
                    cam=combined_cam,
                    target=xai_target,
                    context=context,
                )
            )

            # Smooth after bbox prior.
            combined_cam = self._smooth_cam(
                combined_cam
            )

            combined_cam = (
                self._normalize_cam(
                    combined_cam
                )
            )

        else:

            combined_cam = self._smooth_cam(
                combined_cam
            )

            combined_cam = (
                self._normalize_cam(
                    combined_cam
                )
            )

        heatmap = (
            combined_cam[0]
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        runtime_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        metadata: dict[str, Any] = {
            "input_shape": tuple(
                input_data.shape
            ),
            "output_shape": (
                tuple(output.shape)
                if isinstance(
                    output,
                    torch.Tensor,
                )
                else None
            ),
            "target_class_id": (
                target_class_id
            ),
            "target_score": float(
                target_score.detach().item()
            ),
            "target_task": (
                xai_target.task
                if xai_target is not None
                else "classification"
            ),
            "num_target_layers": len(
                self.target_layers
            ),
            "multi_scale": (
                len(self.target_layers) > 1
            ),
            "spatial_mode": spatial_mode,
        }

        if xai_target is not None:
            metadata.update(
                {
                    "detection_index": xai_target.detection_index,
                    "bbox": xai_target.bbox,
                    "bbox_prior": True,
                    "bbox_prior_model_weight": 0.75,
                    "bbox_prior_geometric_weight": 0.25,
                }
            )

        return ExplanationResult(
            method="grad_cam",
            attribution=heatmap,
            target=target_class_id,
            confidence=confidence,
            runtime_ms=runtime_ms,
            input_type="image",
            metadata=metadata,
        )

    # ============================================================
    # Cleanup
    # ============================================================

    def close(self) -> None:

        for handle in self._forward_handles:
            try:
                handle.remove()
            except Exception:
                pass

        for handle in self._backward_handles:
            try:
                handle.remove()
            except Exception:
                pass

        self._forward_handles.clear()
        self._backward_handles.clear()

        self._activations.clear()
        self._gradients.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass