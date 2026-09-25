import time
from typing import Any, Callable

import numpy as np
import shap
import torch
from skimage.segmentation import slic

from ..core.base import Explainer
from ..core.result import ExplanationResult


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

        self.preprocess = preprocess
        self.nsamples = nsamples
        self.num_segments = num_segments
        self.compactness = compactness
        self.hide_color = hide_color

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

    @staticmethod
    def _extract_target_values(
        shap_values: Any,
        target: int,
        n_segments: int,
    ) -> np.ndarray:
        """
        Normalize SHAP output shapes across versions.
        """

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
        target: int | str | None = None,
    ) -> ExplanationResult:

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