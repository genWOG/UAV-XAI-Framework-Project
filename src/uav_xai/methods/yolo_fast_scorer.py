from __future__ import annotations

from typing import Any, Sequence
import warnings

import numpy as np


class _NMSFreeUnavailable(Exception):
    """Raised when this Ultralytics build cannot disable NMS."""


class YoloDetectionScorer:
    """Batch target-object scores, preferring YOLO26's NMS-free inference.

    The first image batch is evaluated by both the regular predictor and
    ``nms=False``. NMS-free inference is retained only if target-object scores
    agree within ``parity_tolerance``. Unsupported versions and score mismatch
    safely fall back to ordinary model.predict().
    """

    def __init__(
        self,
        model: Any,
        target_bbox: Sequence[float],
        target_class_id: int,
        score_mode: str = "confidence_iou",
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.001,
        parity_tolerance: float = 0.05,
    ) -> None:
        if score_mode not in {"confidence_iou", "thresholded_confidence"}:
            raise ValueError("Unsupported target score mode.")
        if len(target_bbox) != 4:
            raise ValueError("target_bbox must have four coordinates.")
        if not 0.0 <= float(iou_threshold) <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")
        if not 0.0 <= float(parity_tolerance) <= 1.0:
            raise ValueError("parity_tolerance must be between 0 and 1.")

        self.model = model
        self.target_bbox = np.asarray(target_bbox, dtype=np.float32)
        self.target_class_id = int(target_class_id)
        self.score_mode = score_mode
        self.iou_threshold = float(iou_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.parity_tolerance = float(parity_tolerance)
        self.backend = "ultralytics_predict_uninitialized"
        self.max_score_difference: float | None = None
        self._initialized = False
        self._use_nms_free = False

    def __call__(self, bgr_images: Sequence[np.ndarray]) -> np.ndarray:
        images = [np.ascontiguousarray(np.asarray(image)) for image in bgr_images]
        if not images:
            return np.empty((0, 2), dtype=np.float32)

        if not self._initialized:
            standard_results = self._predict(images, nms_free=False)
            standard_scores = self._score_results(standard_results)
            try:
                fast_results = self._predict(images, nms_free=True)
                fast_scores = self._score_results(fast_results)
            except _NMSFreeUnavailable as exc:
                self.backend = "ultralytics_predict_fallback_unsupported"
                self._initialized = True
                warnings.warn(
                    f"NMS-free YOLO inference unavailable ({exc}); "
                    "using ordinary model.predict().",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return standard_scores

            difference = float(
                np.max(np.abs(standard_scores[:, 1] - fast_scores[:, 1]))
            )
            self.max_score_difference = difference
            self._initialized = True
            if difference <= self.parity_tolerance:
                self._use_nms_free = True
                self.backend = "ultralytics_predict_nms_free"
                return fast_scores

            self.backend = "ultralytics_predict_fallback_parity"
            warnings.warn(
                "NMS-free score differs from ordinary predict() by "
                f"{difference:.4f}, above tolerance {self.parity_tolerance:.4f}; "
                "using ordinary model.predict().",
                RuntimeWarning,
                stacklevel=2,
            )
            return standard_scores

        if self._use_nms_free:
            try:
                return self._score_results(self._predict(images, nms_free=True))
            except _NMSFreeUnavailable as exc:
                # A runtime/version-specific failure must not break explanation.
                self._use_nms_free = False
                self.backend = "ultralytics_predict_fallback_runtime"
                warnings.warn(
                    f"NMS-free inference stopped ({exc}); falling back to model.predict().",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return self._score_results(self._predict(images, nms_free=False))

    def _predict(self, images: Sequence[np.ndarray], *, nms_free: bool) -> list[Any]:
        kwargs = {
            "source": list(images),
            "conf": self.confidence_threshold,
            "verbose": False,
        }
        if nms_free:
            kwargs["nms"] = False
        try:
            results = self.model.predict(**kwargs)
        except Exception as exc:
            message = str(exc).lower()
            if nms_free and "nms" in message and any(
                term in message
                for term in ("unexpected", "invalid", "not a valid", "unsupported", "keyword")
            ):
                raise _NMSFreeUnavailable(str(exc)) from exc
            raise
        if results is None or len(results) != len(images):
            raise RuntimeError("Ultralytics returned an unexpected number of results.")
        return list(results)

    def _score_results(self, results: Sequence[Any]) -> np.ndarray:
        scores: list[list[float]] = []
        for result in results:
            best_iou = 0.0
            best_confidence = 0.0
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy
                classes = boxes.cls
                confidences = boxes.conf
                xyxy = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
                classes = classes.detach().cpu().numpy() if hasattr(classes, "detach") else np.asarray(classes)
                confidences = confidences.detach().cpu().numpy() if hasattr(confidences, "detach") else np.asarray(confidences)
                for box, class_id, confidence in zip(xyxy, classes, confidences):
                    if int(class_id) != self.target_class_id:
                        continue
                    overlap = self._bbox_iou(self.target_bbox, box)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_confidence = float(confidence)

            if self.score_mode == "confidence_iou":
                score = best_confidence * best_iou
            else:
                score = best_confidence if best_iou >= self.iou_threshold else 0.0
            scores.append([1.0 - score, score])
        return np.asarray(scores, dtype=np.float32)

    @staticmethod
    def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
        ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
        bx1, by1, bx2, by2 = (float(v) for v in box_b)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0
