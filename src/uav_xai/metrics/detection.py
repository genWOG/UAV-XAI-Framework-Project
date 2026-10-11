from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.prediction import PredictionResult
from ..core.target import XAITarget


@dataclass
class DetectionMetricResult:
    fractions: np.ndarray
    confidences: np.ndarray
    ious: np.ndarray
    auc: float


def _prepare_attribution(
    attribution: np.ndarray,
    height: int,
    width: int,
) -> torch.Tensor:
    tensor = torch.as_tensor(
        attribution,
        dtype=torch.float32,
    )

    if tensor.ndim != 2:
        raise ValueError(
            "Attribution must be a 2D array."
        )

    return F.interpolate(
        tensor[None, None],
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def _iou(
    box_a: list[float],
    box_b: list[float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(
        0.0,
        ay2 - ay1,
    )
    area_b = max(0.0, bx2 - bx1) * max(
        0.0,
        by2 - by1,
    )

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def match_target_detection(
    prediction: PredictionResult,
    target: XAITarget,
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    """
    Match a detection to the original target using class and IoU.

    Returns (confidence, IoU). If no detection reaches the IoU
    threshold, its confidence is treated as zero.
    """

    if target.task != "detection":
        raise ValueError(
            "Detection metrics require a detection target."
        )

    if target.bbox is None:
        raise ValueError(
            "Detection target must contain bbox."
        )

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(
            "iou_threshold must be between 0 and 1."
        )

    best_confidence = 0.0
    best_iou = 0.0

    for detection in prediction.detections or []:
        if detection.class_id != target.class_id:
            continue

        iou = _iou(target.bbox, detection.bbox)

        if iou > best_iou:
            best_iou = iou
            best_confidence = detection.confidence

    if best_iou < iou_threshold:
        return 0.0, float(best_iou)

    return float(best_confidence), float(best_iou)


def _predict_target(
    predictor: Any,
    input_data: Any,
    target: XAITarget,
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    prediction = predictor.predict(input_data)

    return match_target_detection(
        prediction,
        target,
        iou_threshold=iou_threshold,
    )


def compute_detection_deletion(
    predictor: Any,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target: XAITarget,
    steps: int = 20,
    baseline_value: float = 0.0,
    iou_threshold: float = 0.5,
) -> DetectionMetricResult:

    if target.task != "detection":
        raise ValueError(
            "Target must be a detection target."
        )

    if input_tensor.ndim != 4:
        raise ValueError(
            "input_tensor must have shape [B, C, H, W]."
        )

    if input_tensor.shape[0] != 1:
        raise ValueError(
            "Only batch size 1 is supported."
        )

    if steps < 1:
        raise ValueError(
            "steps must be >= 1."
        )

    _, channels, height, width = input_tensor.shape

    attribution_tensor = _prepare_attribution(
        attribution,
        height,
        width,
    )

    sorted_indices = torch.argsort(
        attribution_tensor.flatten(),
        descending=True,
    )

    total_pixels = height * width

    fractions = np.linspace(
        0.0,
        1.0,
        steps + 1,
    )

    confidences = []
    ious = []

    original = input_tensor.detach().clone()

    for fraction in fractions:
        num_pixels = int(
            round(
                fraction * total_pixels
            )
        )

        modified = original.clone()

        if num_pixels > 0:
            indices = sorted_indices[
                :num_pixels
            ]

            flat = modified.view(
                channels,
                -1,
            )

            flat[
                :,
                indices,
            ] = baseline_value

        confidence, iou = _predict_target(
            predictor,
            modified,
            target,
        )

        confidences.append(confidence)
        ious.append(iou)

    confidences = np.asarray(
        confidences,
        dtype=np.float64,
    )
    ious = np.asarray(
        ious,
        dtype=np.float64,
    )

    auc = float(
        np.trapezoid(
            confidences,
            fractions,
        )
    )

    return DetectionMetricResult(
        fractions=fractions,
        confidences=confidences,
        ious=ious,
        auc=auc,
    )


def compute_detection_insertion(
    predictor: Any,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target: XAITarget,
    steps: int = 20,
    baseline_value: float = 0.0,
    iou_threshold: float = 0.5,
) -> DetectionMetricResult:

    if target.task != "detection":
        raise ValueError(
            "Target must be a detection target."
        )

    if input_tensor.ndim != 4:
        raise ValueError(
            "input_tensor must have shape [B, C, H, W]."
        )

    if input_tensor.shape[0] != 1:
        raise ValueError(
            "Only batch size 1 is supported."
        )

    if steps < 1:
        raise ValueError(
            "steps must be >= 1."
        )

    _, channels, height, width = input_tensor.shape

    attribution_tensor = _prepare_attribution(
        attribution,
        height,
        width,
    )

    sorted_indices = torch.argsort(
        attribution_tensor.flatten(),
        descending=True,
    )

    total_pixels = height * width

    fractions = np.linspace(
        0.0,
        1.0,
        steps + 1,
    )

    confidences = []
    ious = []

    for fraction in fractions:
        num_pixels = int(
            round(
                fraction * total_pixels
            )
        )

        modified = torch.full_like(
            input_tensor,
            fill_value=baseline_value,
        )

        if num_pixels > 0:
            indices = sorted_indices[
                :num_pixels
            ]

            modified_flat = modified.view(
                channels,
                -1,
            )

            original_flat = input_tensor.view(
                channels,
                -1,
            )

            modified_flat[
                :,
                indices,
            ] = original_flat[
                :,
                indices,
            ]

        confidence, iou = _predict_target(
            predictor,
            modified,
            target,
            iou_threshold=iou_threshold,
        )

        confidences.append(confidence)
        ious.append(iou)

    confidences = np.asarray(
        confidences,
        dtype=np.float64,
    )
    ious = np.asarray(
        ious,
        dtype=np.float64,
    )

    auc = float(
        np.trapezoid(
            confidences,
            fractions,
        )
    )

    return DetectionMetricResult(
        fractions=fractions,
        confidences=confidences,
        ious=ious,
        auc=auc,
    )