from dataclasses import dataclass, field
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
    class_confidences: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    best_iou_candidate_confidences: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    has_target_class_detection: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_)
    )
    target_detection_matched: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_)
    )

    @property
    def detection_scores(self) -> np.ndarray:
        """Legacy IoU-gated scores used for the existing AUC."""
        return self.confidences

    @property
    def localization_scores(self) -> np.ndarray:
        """Continuous confidence x IoU score, independent of the match cutoff."""
        return (
            np.clip(self.best_iou_candidate_confidences, 0.0, 1.0)
            * np.clip(self.ious, 0.0, 1.0)
        )

    @property
    def localization_auc(self) -> float:
        """AUC for the continuous localization-weighted score."""
        return float(np.trapezoid(self.localization_scores, self.fractions))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete per-step curve in JSON-compatible form."""
        return {
            "fractions": self.fractions.tolist(),
            "detection_scores": self.detection_scores.tolist(),
            "localization_scores": self.localization_scores.tolist(),
            "localization_auc": self.localization_auc,
            "class_confidences": self.class_confidences.tolist(),
            "best_iou_candidate_confidences": self.best_iou_candidate_confidences.tolist(),
            "ious": self.ious.tolist(),
            "has_target_class_detection": (
                self.has_target_class_detection.tolist()
            ),
            "target_detection_matched": (
                self.target_detection_matched.tolist()
            ),
        }


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


@dataclass(frozen=True)
class _TargetDetectionAssessment:
    detection_score: float
    class_confidence: float
    matched_confidence: float
    iou: float
    has_target_class_detection: bool
    target_detection_matched: bool


def _inspect_target_detection(
    prediction: PredictionResult,
    target: XAITarget,
    iou_threshold: float = 0.5,
) -> _TargetDetectionAssessment:
    """Separate class confidence, target localization and legacy score."""
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

    # Keep the old score path separate to preserve its strict-greater tie handling.
    legacy_best_confidence = 0.0
    legacy_best_iou = 0.0

    # Diagnostic candidate: highest IoU among same-class detections, even below threshold.
    best_iou = 0.0
    matched_confidence = 0.0
    has_iou_candidate = False
    class_confidence = 0.0
    has_target_class_detection = False

    for detection in prediction.detections or []:
        if detection.class_id != target.class_id:
            continue

        confidence = float(detection.confidence)
        has_target_class_detection = True
        class_confidence = max(class_confidence, confidence)
        iou = _iou(target.bbox, detection.bbox)

        if not has_iou_candidate or iou > best_iou:
            best_iou = iou
            matched_confidence = confidence
            has_iou_candidate = True

        if iou > legacy_best_iou:
            legacy_best_iou = iou
            legacy_best_confidence = confidence

    # Exact legacy semantics: score is zero when the best same-class IoU is below threshold.
    detection_score = (
        0.0
        if legacy_best_iou < iou_threshold
        else legacy_best_confidence
    )

    return _TargetDetectionAssessment(
        detection_score=float(detection_score),
        class_confidence=float(class_confidence),
        matched_confidence=float(matched_confidence),
        iou=float(best_iou),
        has_target_class_detection=has_target_class_detection,
        target_detection_matched=(
            has_target_class_detection
            and best_iou > 0.0
            and best_iou >= iou_threshold
        ),
    )


def match_target_detection(
    prediction: PredictionResult,
    target: XAITarget,
    iou_threshold: float = 0.5,
) -> tuple[float, float]:
    """Return the legacy IoU-gated score and best same-class IoU."""
    assessment = _inspect_target_detection(
        prediction,
        target,
        iou_threshold=iou_threshold,
    )
    return assessment.detection_score, assessment.iou


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


def _predict_target_assessment(
    predictor: Any,
    input_data: Any,
    target: XAITarget,
    iou_threshold: float = 0.5,
) -> _TargetDetectionAssessment:
    prediction = predictor.predict(input_data)
    return _inspect_target_detection(
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
    class_confidences = []
    best_iou_candidate_confidences = []
    has_target_class_detections = []
    target_detection_matches = []

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

        assessment = _predict_target_assessment(
            predictor=predictor,
            input_data=modified,
            target=target,
            iou_threshold=iou_threshold,
        )

        confidences.append(assessment.detection_score)
        ious.append(assessment.iou)
        class_confidences.append(assessment.class_confidence)
        best_iou_candidate_confidences.append(assessment.matched_confidence)
        has_target_class_detections.append(
            assessment.has_target_class_detection
        )
        target_detection_matches.append(
            assessment.target_detection_matched
        )

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
        class_confidences=np.asarray(class_confidences, dtype=np.float64),
        best_iou_candidate_confidences=np.asarray(best_iou_candidate_confidences, dtype=np.float64),
        has_target_class_detection=np.asarray(
            has_target_class_detections, dtype=np.bool_
        ),
        target_detection_matched=np.asarray(
            target_detection_matches, dtype=np.bool_
        ),
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
    class_confidences = []
    best_iou_candidate_confidences = []
    has_target_class_detections = []
    target_detection_matches = []

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

        assessment = _predict_target_assessment(
            predictor=predictor,
            input_data=modified,
            target=target,
            iou_threshold=iou_threshold,
        )

        confidences.append(assessment.detection_score)
        ious.append(assessment.iou)
        class_confidences.append(assessment.class_confidence)
        best_iou_candidate_confidences.append(assessment.matched_confidence)
        has_target_class_detections.append(
            assessment.has_target_class_detection
        )
        target_detection_matches.append(
            assessment.target_detection_matched
        )

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
        class_confidences=np.asarray(class_confidences, dtype=np.float64),
        best_iou_candidate_confidences=np.asarray(best_iou_candidate_confidences, dtype=np.float64),
        has_target_class_detection=np.asarray(
            has_target_class_detections, dtype=np.bool_
        ),
        target_detection_matched=np.asarray(
            target_detection_matches, dtype=np.bool_
        ),
    )
