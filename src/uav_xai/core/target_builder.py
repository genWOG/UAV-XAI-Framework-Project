from typing import Any

import torch

from .target import XAITarget


class TargetBuilder:
    """
    Build differentiable scalar targets for XAI methods.

    Supports:
        - classification;
        - Ultralytics YOLO26 detection.

    For YOLO detection:
        - use the actual Detect decoding logic;
        - match the requested bbox against decoded one2one boxes;
        - use sigmoid class scores;
        - build a differentiable target from relevant queries.
    """

    # ============================================================
    # Public API
    # ============================================================

    @staticmethod
    def build(
        model: Any,
        input_data: Any,
        target: XAITarget,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:

        context = context or {}

        if target.task == "classification":
            return TargetBuilder._build_classification(
                model=model,
                input_data=input_data,
                target=target,
            )

        if target.task == "detection":
            return TargetBuilder._build_detection(
                model=model,
                input_data=input_data,
                target=target,
                context=context,
            )

        raise ValueError(
            f"Unsupported target task: {target.task}"
        )

    # ============================================================
    # Classification
    # ============================================================

    @staticmethod
    def _build_classification(
        model: Any,
        input_data: torch.Tensor,
        target: XAITarget,
    ) -> torch.Tensor:

        if not isinstance(
            input_data,
            torch.Tensor,
        ):
            raise TypeError(
                "Classification XAI requires "
                "a torch.Tensor input."
            )

        output = model(
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
                "Classification output must have shape "
                "[batch, classes]."
            )

        if not (
            0 <= target.class_id
            < output.shape[1]
        ):
            raise IndexError(
                f"Class ID {target.class_id} "
                f"is out of range for "
                f"{output.shape[1]} classes."
            )

        score = output[
            0,
            target.class_id,
        ]

        if not score.requires_grad:
            raise RuntimeError(
                "Classification target is not differentiable."
            )

        return score

    # ============================================================
    # Detection network
    # ============================================================

    @staticmethod
    def _get_detection_network(
        model: Any,
    ) -> torch.nn.Module:

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
            network = wrapper_model

        else:
            network = model

        if not isinstance(
            network,
            torch.nn.Module,
        ):
            raise TypeError(
                "Could not resolve a PyTorch detection model."
            )

        return network

    # ============================================================
    # IoU
    # ============================================================

    @staticmethod
    def _box_iou(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor,
    ) -> torch.Tensor:

        if (
            boxes1.ndim != 2
            or boxes1.shape[1] != 4
        ):
            raise ValueError(
                "boxes1 must have shape [N, 4]."
            )

        if (
            boxes2.ndim != 2
            or boxes2.shape[1] != 4
        ):
            raise ValueError(
                "boxes2 must have shape [M, 4]."
            )

        top_left = torch.maximum(
            boxes1[:, None, :2],
            boxes2[None, :, :2],
        )

        bottom_right = torch.minimum(
            boxes1[:, None, 2:],
            boxes2[None, :, 2:],
        )

        wh = torch.clamp(
            bottom_right - top_left,
            min=0.0,
        )

        intersection = (
            wh[..., 0]
            * wh[..., 1]
        )

        area1 = (
            torch.clamp(
                boxes1[:, 2]
                - boxes1[:, 0],
                min=0.0,
            )
            * torch.clamp(
                boxes1[:, 3]
                - boxes1[:, 1],
                min=0.0,
            )
        )

        area2 = (
            torch.clamp(
                boxes2[:, 2]
                - boxes2[:, 0],
                min=0.0,
            )
            * torch.clamp(
                boxes2[:, 3]
                - boxes2[:, 1],
                min=0.0,
            )
        )

        union = (
            area1[:, None]
            + area2[None, :]
            - intersection
            + 1e-8
        )

        return (
            intersection
            / union
        )

    # ============================================================
    # Detect head
    # ============================================================

    @staticmethod
    def _get_detect_head(
        network: torch.nn.Module,
    ) -> torch.nn.Module:

        layers = getattr(
            network,
            "model",
            None,
        )

        if layers is None:
            raise ValueError(
                "DetectionModel does not expose "
                "its layer list."
            )

        if len(layers) == 0:
            raise ValueError(
                "DetectionModel layer list is empty."
            )

        detect_head = layers[-1]

        if not hasattr(
            detect_head,
            "_get_decode_boxes",
        ):
            raise TypeError(
                "Final YOLO layer does not expose "
                "_get_decode_boxes()."
            )

        return detect_head

    # ============================================================
    # Detection
    # ============================================================

    @staticmethod
    def _build_detection(
        model: Any,
        input_data: torch.Tensor,
        target: XAITarget,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:

        context = context or {}

        if not isinstance(
            input_data,
            torch.Tensor,
        ):
            raise TypeError(
                "Detection XAI requires a torch.Tensor input."
            )

        if input_data.ndim != 4:
            raise ValueError(
                "Detection XAI input must have shape "
                "[batch, channels, height, width]."
            )

        if target.bbox is None:
            raise ValueError(
                "Detection XAI requires target.bbox."
            )

        network = (
            TargetBuilder._get_detection_network(
                model
            )
        )

        network.eval()

        # --------------------------------------------------------
        # Ensure autograd-compatible input.
        # --------------------------------------------------------

        if not input_data.requires_grad:
            input_data = (
                input_data
                .detach()
                .clone()
                .requires_grad_(True)
            )

        with torch.enable_grad():

            # ----------------------------------------------------
            # Run the actual DetectionModel.
            # ----------------------------------------------------

            raw_output = network(
                input_data
            )

            if not (
                isinstance(
                    raw_output,
                    tuple,
                )
                and len(raw_output) == 2
            ):
                raise ValueError(
                    "Expected YOLO DetectionModel "
                    "to return (output, predictions)."
                )

            _, predictions = raw_output

            if not isinstance(
                predictions,
                dict,
            ):
                raise TypeError(
                    "YOLO predictions must be a dictionary."
                )

            one2one = predictions.get(
                "one2one"
            )

            if not isinstance(
                one2one,
                dict,
            ):
                raise ValueError(
                    "YOLO one2one predictions are missing."
                )

            raw_scores = one2one.get(
                "scores"
            )

            if not isinstance(
                raw_scores,
                torch.Tensor,
            ):
                raise TypeError(
                    "YOLO one2one scores must be a Tensor."
                )

            if raw_scores.ndim != 3:
                raise ValueError(
                    "YOLO one2one scores must have shape "
                    "[batch, classes, queries]."
                )

            # ----------------------------------------------------
            # CRITICAL:
            #
            # Do NOT manually decode one2one["boxes"].
            #
            # Ultralytics itself performs:
            #
            #   anchors
            #   + stride
            #   + decode_bboxes
            #
            # through _get_decode_boxes().
            # ----------------------------------------------------

            network_detect = (
                TargetBuilder._get_detect_head(
                    network
                )
            )

            decoded_boxes = (
                network_detect._get_decode_boxes(
                    one2one
                )
            )

            if not isinstance(
                decoded_boxes,
                torch.Tensor,
            ):
                raise TypeError(
                    "YOLO decoded boxes must be a Tensor."
                )

            if decoded_boxes.ndim != 3:
                raise ValueError(
                    "YOLO decoded boxes must have shape "
                    "[batch, 4, queries]."
                )

            # ----------------------------------------------------
            # YOLO26 end-to-end decoding returns xyxy boxes
            # in model-input coordinates.
            # ----------------------------------------------------

            candidate_boxes = (
                decoded_boxes[
                    0
                ]
                .permute(
                    1,
                    0,
                )
                .contiguous()
            )

            if candidate_boxes.shape[1] != 4:
                raise ValueError(
                    "Decoded YOLO boxes must contain "
                    "4 coordinates."
                )

            # ----------------------------------------------------
            # Convert requested original-image bbox to
            # letterboxed model-input coordinates.
            # ----------------------------------------------------

            letterbox = context.get(
                "letterbox"
            )

            if letterbox is None:
                raise ValueError(
                    "Detection XAI requires letterbox metadata."
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
                float(value)
                for value in target.bbox
            )

            target_box = torch.tensor(
                [
                    x1 * scale + pad_x,
                    y1 * scale + pad_y,
                    x2 * scale + pad_x,
                    y2 * scale + pad_y,
                ],
                dtype=candidate_boxes.dtype,
                device=candidate_boxes.device,
            ).unsqueeze(0)

            print(
                "TARGET BOX:",
                target_box.detach().cpu(),
            )

            # ----------------------------------------------------
            # Match decoded boxes to requested object.
            # ----------------------------------------------------

            ious = TargetBuilder._box_iou(
                candidate_boxes,
                target_box,
            )[:, 0]

            # ----------------------------------------------------
            # Class probability.
            # ----------------------------------------------------

            if not (
                0 <= target.class_id
                < raw_scores.shape[1]
            ):
                raise IndexError(
                    f"Class ID {target.class_id} "
                    f"is out of range for "
                    f"{raw_scores.shape[1]} classes."
                )

            class_logits = raw_scores[
                0,
                target.class_id,
            ]

            class_probabilities = torch.sigmoid(
                class_logits
            )

            # ----------------------------------------------------
            # Candidate selection.
            # ----------------------------------------------------

            iou_threshold = float(
                context.get(
                    "detection_iou_threshold",
                    0.10,
                )
            )

            score_threshold = float(
                context.get(
                    "detection_score_threshold",
                    0.01,
                )
            )

            top_k = int(
                context.get(
                    "detection_top_k",
                    16,
                )
            )

            candidate_mask = (
                (ious >= iou_threshold)
                & (
                    class_probabilities
                    >= score_threshold
                )
            )

            candidate_indices = torch.nonzero(
                candidate_mask,
                as_tuple=False,
            ).flatten()

            # ----------------------------------------------------
            # Fallback: rank by IoU.
            # ----------------------------------------------------

            if candidate_indices.numel() == 0:

                top_count = min(
                    top_k,
                    ious.numel(),
                )

                candidate_indices = torch.topk(
                    ious,
                    k=top_count,
                    largest=True,
                ).indices

            else:

                relevance = (
                    ious[
                        candidate_indices
                    ].detach()
                    * class_probabilities[
                        candidate_indices
                    ].detach()
                )

                top_count = min(
                    top_k,
                    candidate_indices.numel(),
                )

                selected = torch.topk(
                    relevance,
                    k=top_count,
                    largest=True,
                ).indices

                candidate_indices = (
                    candidate_indices[
                        selected
                    ]
                )

            # ----------------------------------------------------
            # Selected queries.
            # ----------------------------------------------------

            selected_ious = ious[
                candidate_indices
            ]

            selected_probabilities = (
                class_probabilities[
                    candidate_indices
                ]
            )

            selected_logits = class_logits[
                candidate_indices
            ]

            # ----------------------------------------------------
            # Build differentiable target.
            #
            # Weight by IoU and class probability, but detach
            # those weights so the gradient flows only through
            # the detector score itself.
            # ----------------------------------------------------

            weights = (
                selected_ious.detach()
                * selected_probabilities.detach()
            )

            if float(
                weights.sum()
                .detach()
                .item()
            ) <= 1e-8:

                weights = torch.ones_like(
                    weights
                )

            weights = (
                weights
                / weights.sum()
            )

            target_score = (
                weights
                * selected_logits
            ).sum()

            # ----------------------------------------------------
            # Diagnostics.
            # ----------------------------------------------------

            best_local = int(
                torch.argmax(
                    selected_ious
                ).item()
            )

            best_query = int(
                candidate_indices[
                    best_local
                ].item()
            )

            best_iou = float(
                selected_ious[
                    best_local
                ]
                .detach()
                .item()
            )

            print(
                "MATCHED QUERIES:",
                candidate_indices
                .detach()
                .cpu()
                .tolist(),
            )

            print(
                "MATCHED IoUs:",
                [
                    round(
                        float(value),
                        4,
                    )
                    for value in (
                        selected_ious
                        .detach()
                        .cpu()
                        .tolist()
                    )
                ],
            )

            print(
                "MATCHED CLASS PROBS:",
                [
                    round(
                        float(value),
                        4,
                    )
                    for value in (
                        selected_probabilities
                        .detach()
                        .cpu()
                        .tolist()
                    )
                ],
            )

            print(
                "BEST QUERY:",
                best_query,
                "IOU:",
                best_iou,
            )

            print(
                "TARGET SCORE:",
                float(
                    target_score
                    .detach()
                    .item()
                ),
            )

            print(
                "TARGET PROBABILITY:",
                float(
                    torch.sigmoid(
                        target_score
                    )
                    .detach()
                    .item()
                ),
            )

            if not target_score.requires_grad:
                raise RuntimeError(
                    "YOLO detection target "
                    "is not differentiable."
                )

            return target_score