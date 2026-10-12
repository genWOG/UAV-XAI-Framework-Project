import time
from typing import Any, Callable

import numpy as np
import cv2
import shap
import torch
from skimage.segmentation import slic

from ..core.base import Explainer
from ..core.result import ExplanationResult
from ..core.target import XAITarget


class ShapImageExplainer(Explainer):
    """
    SHAP explainer for image classification.

    The image is represented by superpixels. SHAP estimates
    the contribution of each superpixel to the selected class,
    then the superpixel values are projected back to a 2D
    attribution map.

    Parameters
    ----------
    preprocess:
        Converts an RGB HWC numpy image to a model input tensor
        with shape [C, H, W].

    nsamples:
        Number of SHAP samples used by KernelExplainer.

    num_segments:
        Approximate number of superpixels.

    compactness:
        Compactness parameter for SLIC segmentation.

    hide_color:
        RGB baseline used for hidden superpixels.
    """

    def __init__(
        self,
        preprocess: Callable[[np.ndarray], torch.Tensor],
        nsamples: int = 100,
        num_segments: int = 30,
        compactness: float = 10.0,
        hide_color: int | float = 0,
        target_iou_threshold: float = 0.5,
        baseline_mode: str = "constant",
        prediction_batch_size: int = 32,
        target_score_mode: str = "confidence_iou",
        explanation_image_size: int | None = None,
        algorithm: str = "kernel",
    ) -> None:

        if nsamples < 1:
            raise ValueError(
                "nsamples must be >= 1."
            )

        if num_segments < 2:
            raise ValueError(
                "num_segments must be >= 2."
            )

        if compactness <= 0:
            raise ValueError(
                "compactness must be > 0."
            )

        if not 0.0 <= target_iou_threshold <= 1.0:
            raise ValueError(
                "target_iou_threshold must be between 0 and 1."
            )

        self.preprocess = preprocess
        self.nsamples = nsamples
        self.num_segments = num_segments
        self.compactness = compactness
        self.hide_color = hide_color
        self.target_iou_threshold = target_iou_threshold
        if baseline_mode not in {"constant", "segment_mean"}:
            raise ValueError("baseline_mode must be 'constant' or 'segment_mean'.")
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be >= 1.")
        if target_score_mode not in {"confidence_iou", "thresholded_confidence"}:
            raise ValueError("target_score_mode must be 'confidence_iou' or 'thresholded_confidence'.")
        self.baseline_mode = baseline_mode
        self.prediction_batch_size = int(prediction_batch_size)
        self.target_score_mode = target_score_mode
        if explanation_image_size is not None and int(explanation_image_size) < 64:
            raise ValueError("explanation_image_size must be None or >= 64.")
        self.explanation_image_size = (
            None if explanation_image_size is None else int(explanation_image_size)
        )
        if algorithm not in {"kernel", "partition"}:
            raise ValueError("algorithm must be 'kernel' or 'partition'.")
        self.algorithm = algorithm

    def _validate_image(
        self,
        input_data: Any,
    ) -> np.ndarray:

        image = np.asarray(
            input_data
        )

        if image.ndim != 3:
            raise ValueError(
                "Image must have shape [H, W, 3]."
            )

        if image.shape[2] != 3:
            raise ValueError(
                "Only RGB images with 3 channels are supported."
            )

        if not np.isfinite(
            image.astype(np.float32)
        ).all():
            raise ValueError(
                "Image contains NaN or Inf values."
            )

        if image.dtype != np.uint8:

            image_min = float(
                image.min()
            )
            image_max = float(
                image.max()
            )

            if (
                image_min >= 0.0
                and image_max <= 1.0
            ):
                image = (
                    image * 255.0
                ).clip(
                    0,
                    255,
                )
            else:
                image = image.clip(
                    0,
                    255,
                )

            image = image.astype(
                np.uint8
            )

        return image

    def _build_predict_fn(
        self,
        model: Any,
        image: np.ndarray,
        segments: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Build a prediction function operating on binary
        superpixel masks.
        """

        n_segments = int(
            segments.max()
        ) + 1

        baseline = np.full_like(
            image,
            fill_value=self.hide_color,
        )

        def predict_fn(
            masks: np.ndarray,
        ) -> np.ndarray:

            masks = np.asarray(
                masks,
                dtype=np.float32,
            )

            if masks.ndim == 1:
                masks = masks.reshape(
                    1,
                    -1,
                )

            if masks.ndim != 2:
                raise ValueError(
                    "SHAP masks must have shape "
                    "[samples, segments]."
                )

            if masks.shape[1] != n_segments:
                raise ValueError(
                    "Number of mask features does not "
                    "match the number of segments."
                )

            tensors = []

            for mask in masks:

                perturbed = baseline.copy()

                for segment_id in range(
                    n_segments
                ):
                    if mask[segment_id] > 0.5:
                        perturbed[
                            segments == segment_id
                        ] = image[
                            segments == segment_id
                        ]

                tensor = self.preprocess(
                    perturbed
                )

                if not isinstance(
                    tensor,
                    torch.Tensor,
                ):
                    raise TypeError(
                        "preprocess must return torch.Tensor."
                    )

                if tensor.ndim != 3:
                    raise ValueError(
                        "preprocess must return tensor "
                        "with shape [C, H, W]."
                    )

                tensors.append(
                    tensor
                )

            batch = torch.stack(
                tensors,
                dim=0,
            )

            try:
                device = next(
                    model.parameters()
                ).device

                batch = batch.to(
                    device
                )

            except (
                AttributeError,
                StopIteration,
            ):
                pass

            with torch.no_grad():

                output = model(
                    batch
                )

                if output.ndim != 2:
                    raise ValueError(
                        "Model output must have shape "
                        "[batch_size, num_classes]."
                    )

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )

            return (
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

        return predict_fn

    def _resize_for_explanation(self, image: np.ndarray) -> np.ndarray:
        """Downscale the perturbation workspace, preserving aspect ratio."""
        limit = self.explanation_image_size
        if limit is None or max(image.shape[:2]) <= limit:
            return image
        scale = limit / float(max(image.shape[:2]))
        width = max(1, int(round(image.shape[1] * scale)))
        height = max(1, int(round(image.shape[0] * scale)))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _bbox_iou(
        box_a: list[float],
        box_b: list[float],
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0.0 else 0.0

    def _build_detection_predict_fn(
        self,
        model: Any,
        target: XAITarget,
        target_bbox: list[float] | None = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        if not hasattr(model, "predict"):
            raise TypeError("Detection SHAP requires a model with predict().")
        if target.bbox is None:
            raise ValueError("Detection SHAP requires target.bbox.")

        target_bbox = [
            float(value) for value in (
                target.bbox if target_bbox is None else target_bbox
            )
        ]

        from .yolo_fast_scorer import YoloDetectionScorer
        scorer = YoloDetectionScorer(
            model=model,
            target_bbox=target_bbox,
            target_class_id=target.class_id,
            score_mode=self.target_score_mode,
            iou_threshold=self.target_iou_threshold,
            confidence_threshold=0.001,
        )
        def predict_images(images: np.ndarray) -> np.ndarray:
            rgb_images = [self._validate_image(image) for image in images]
            bgr_images = [image[:, :, ::-1].copy() for image in rgb_images]
            return scorer(bgr_images)

        predict_images.scorer = scorer
        return predict_images

    def _explain_detection(
        self,
        model: Any,
        input_data: Any,
        target: XAITarget,
    ) -> ExplanationResult:
        start_time = time.perf_counter()
        image = self._validate_image(input_data)
        working_image = self._resize_for_explanation(image)
        scale_x = working_image.shape[1] / image.shape[1]
        scale_y = working_image.shape[0] / image.shape[0]
        target_bbox = [
            float(target.bbox[0]) * scale_x,
            float(target.bbox[1]) * scale_y,
            float(target.bbox[2]) * scale_x,
            float(target.bbox[3]) * scale_y,
        ]
        segments = slic(
            working_image,
            n_segments=self.num_segments,
            compactness=self.compactness,
            start_label=0,
            channel_axis=-1,
        )
        n_segments = int(segments.max()) + 1
        detection_predict = self._build_detection_predict_fn(
            model, target, target_bbox=target_bbox
        )
        if self.baseline_mode == "segment_mean":
            baseline = np.empty_like(working_image)
            for segment_id in range(n_segments):
                segment_mask = segments == segment_id
                mean_color = np.rint(
                    working_image[segment_mask].mean(axis=0)
                ).clip(0, 255).astype(working_image.dtype)
                baseline[segment_mask] = mean_color
        else:
            baseline = np.full_like(working_image, fill_value=self.hide_color)

        def predict_fn(masks: np.ndarray) -> np.ndarray:
            masks = np.asarray(masks, dtype=np.float32)
            if masks.ndim == 1:
                masks = masks.reshape(1, -1)
            if masks.ndim != 2 or masks.shape[1] != n_segments:
                raise ValueError(
                    "SHAP masks must have shape [samples, number_of_segments]."
                )
            score_batches = []
            for batch_start in range(0, len(masks), self.prediction_batch_size):
                mask_batch = masks[batch_start:batch_start + self.prediction_batch_size]
                visible_pixels = mask_batch[:, segments] > 0.5
                perturbed_batch = np.where(
                    visible_pixels[..., np.newaxis],
                    working_image[np.newaxis, ...],
                    baseline[np.newaxis, ...],
                )
                score_batches.append(detection_predict(perturbed_batch))
            if not score_batches:
                return np.empty((0, 2), dtype=np.float32)
            return np.concatenate(score_batches, axis=0)

        all_on = np.ones((1, n_segments), dtype=np.float32)
        original_scores = predict_fn(all_on)
        background = np.zeros((1, n_segments), dtype=np.float32)
        partition_max_evals = None
        if self.algorithm == "partition":
            from scipy.cluster.hierarchy import linkage

            # Image-aware hierarchy over superpixels: spatial centroids plus mean RGB.
            yy, xx = np.indices(segments.shape)
            descriptors = []
            for segment_id in range(n_segments):
                selected = segments == segment_id
                descriptors.append([
                    float(xx[selected].mean()) / max(1, segments.shape[1] - 1),
                    float(yy[selected].mean()) / max(1, segments.shape[0] - 1),
                    *(working_image[selected].mean(axis=0).astype(np.float64) / 255.0),
                ])
            clustering = linkage(np.asarray(descriptors, dtype=np.float64), method="ward")
            masker = shap.maskers.Partition(data=background, clustering=clustering)
            explainer = shap.PartitionExplainer(predict_fn, masker)
            partition_max_evals = max(self.nsamples, 2 * n_segments + 1)
            shap_result = explainer(
                all_on,
                max_evals=partition_max_evals,
                batch_size=self.prediction_batch_size,
                fixed_context=0,
                silent=True,
            )
            shap_values = shap_result.values
        else:
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(all_on, nsamples=self.nsamples)

        segment_attribution = self._extract_target_values(
            shap_values=shap_values,
            target=1,
            n_segments=n_segments,
        )
        attribution_work = np.zeros(segments.shape, dtype=np.float32)
        for segment_id, value in enumerate(segment_attribution):
            attribution_work[segments == segment_id] = value
        if attribution_work.shape != image.shape[:2]:
            attribution = cv2.resize(
                attribution_work,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32, copy=False)
        else:
            attribution = attribution_work

        confidence = float(original_scores[0, 1])
        runtime_ms = (time.perf_counter() - start_time) * 1000.0
        return ExplanationResult(
            method="shap",
            attribution=attribution,
            target=target.class_id,
            confidence=confidence,
            runtime_ms=runtime_ms,
            input_type="image",
            feature_names=None,
            metadata={
                "nsamples": self.nsamples,
                "num_segments": n_segments,
                "compactness": self.compactness,
                "image_shape": list(image.shape),
                "explanation_image_shape": list(working_image.shape),
                "explanation_image_size": self.explanation_image_size,
                "explainer": "shap_image",
                "algorithm": self.algorithm,
                "partition_max_evals": partition_max_evals,
                "segments": segments,
                "segment_attribution": segment_attribution,
                "target_task": "detection",
                "target_class_id": target.class_id,
                "detection_index": target.detection_index,
                "bbox": list(target.bbox),
                "target_iou_threshold": self.target_iou_threshold,
                "baseline_mode": self.baseline_mode,
                "prediction_batch_size": self.prediction_batch_size,
                "prediction_backend": detection_predict.scorer.backend,
                "fast_path_max_score_difference": detection_predict.scorer.max_score_difference,
                "target_score_mode": self.target_score_mode,
            },
        )

    @staticmethod
    def _extract_target_values(
        shap_values: Any,
        target: int,
        n_segments: int,
    ) -> np.ndarray:
        """
        Normalize SHAP output shapes across versions.
        """

        if hasattr(shap_values, "values"):
            shap_values = shap_values.values

        if isinstance(
            shap_values,
            list,
        ):
            values = np.asarray(
                shap_values[target]
            )

        else:
            values = np.asarray(
                shap_values
            )

            if (
                values.ndim == 3
                and values.shape[-1] > 1
            ):
                values = values[
                    :,
                    :,
                    target,
                ]

        if values.ndim == 2:
            values = values[0]

        if values.ndim != 1:
            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{values.shape}"
            )

        if len(values) != n_segments:
            raise ValueError(
                "Number of SHAP values does not match "
                "number of segments."
            )

        return values.astype(
            np.float32
        )

    def explain(
        self,
        model: Any,
        input_data: Any,
        target: int | str | XAITarget | None = None,
    ) -> ExplanationResult:

        if isinstance(target, XAITarget):
            if target.task != "detection":
                raise ValueError("SHAP received an unsupported XAI target.")
            return self._explain_detection(model=model, input_data=input_data, target=target)

        start_time = time.perf_counter()

        image = self._validate_image(
            input_data
        )

        segments = slic(
            image,
            n_segments=self.num_segments,
            compactness=self.compactness,
            start_label=0,
            channel_axis=-1,
        )

        n_segments = int(
            segments.max()
        ) + 1

        # Predict original image.
        original_predict = (
            self._build_predict_fn(
                model=model,
                image=image,
                segments=segments,
            )
        )

        all_on = np.ones(
            (1, n_segments),
            dtype=np.float32,
        )

        probabilities = original_predict(
            all_on
        )

        if target is None:
            target = int(
                np.argmax(
                    probabilities[0]
                )
            )

        target = int(target)

        if not (
            0 <= target < probabilities.shape[1]
        ):
            raise ValueError(
                f"Target class {target} is out of range."
            )

        # Predict function used by SHAP.
        predict_fn = self._build_predict_fn(
            model=model,
            image=image,
            segments=segments,
        )

        background = np.zeros(
            (1, n_segments),
            dtype=np.float32,
        )

        shap_explainer = shap.KernelExplainer(
            predict_fn,
            background,
        )

        shap_values = (
            shap_explainer.shap_values(
                all_on,
                nsamples=self.nsamples,
            )
        )

        segment_attribution = (
            self._extract_target_values(
                shap_values=shap_values,
                target=target,
                n_segments=n_segments,
            )
        )

        attribution = np.zeros(
            segments.shape,
            dtype=np.float32,
        )

        for segment_id, value in enumerate(
            segment_attribution
        ):
            attribution[
                segments == segment_id
            ] = value

        confidence = float(
            probabilities[
                0,
                target,
            ]
        )

        runtime_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        return ExplanationResult(
            method="shap",
            attribution=attribution,
            target=target,
            confidence=confidence,
            runtime_ms=runtime_ms,
            input_type="image",
            feature_names=None,
            metadata={
                "nsamples": self.nsamples,
                "num_segments": n_segments,
                "compactness": self.compactness,
                "image_shape": list(
                    image.shape
                ),
                "explainer": "shap_image",
                "segments": segments,
                "segment_attribution": (
                    segment_attribution
                ),
            },
        )
