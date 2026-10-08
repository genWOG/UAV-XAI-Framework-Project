import time
from typing import Any, Callable

import numpy as np
import shap

from ..core.base import Explainer
from ..core.result import ExplanationResult


class ShapTabularExplainer(Explainer):
    """
    SHAP explainer for tabular data.

    Designed for black-box classification models exposing
    ``predict_proba(X)``.
    """

    def __init__(
        self,
        background_data: np.ndarray,
        feature_names: list[str],
        nsamples: int = 100,
    ) -> None:

        background_data = np.asarray(
            background_data,
            dtype=np.float32,
        )

        if background_data.ndim != 2:
            raise ValueError(
                "background_data must be a 2D array."
            )

        if background_data.shape[1] != len(
            feature_names
        ):
            raise ValueError(
                "Number of feature_names must match "
                "background_data columns."
            )

        if nsamples < 1:
            raise ValueError(
                "nsamples must be >= 1."
            )

        self.feature_names = feature_names
        self.nsamples = nsamples

        self._background_data = (
            background_data
        )

    def _get_predict_fn(
        self,
        model: Any,
    ) -> Callable[[np.ndarray], np.ndarray]:

        if hasattr(model, "predict_proba"):
            return model.predict_proba

        if callable(model):
            return model

        raise TypeError(
            "Model must provide predict_proba(X) "
            "or be callable."
        )

    def explain(
        self,
        model: Any,
        input_data: Any,
        target: int | str | None = None,
    ) -> ExplanationResult:

        start_time = time.perf_counter()

        instance = np.asarray(
            input_data,
            dtype=np.float32,
        )

        if instance.ndim == 1:
            instance = instance.reshape(
                1,
                -1,
            )

        if instance.ndim != 2:
            raise ValueError(
                "input_data must have shape "
                "[features] or [1, features]."
            )

        if instance.shape[0] != 1:
            raise ValueError(
                "Only one sample can be explained at a time."
            )

        if instance.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                "Number of input features must match "
                "feature_names."
            )

        predict_fn = self._get_predict_fn(
            model
        )

        probabilities = predict_fn(
            instance
        )

        if probabilities.ndim != 2:
            raise ValueError(
                "predict_proba must return a 2D array."
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

        explainer = shap.KernelExplainer(
            predict_fn,
            self._background_data,
        )

        shap_values = explainer.shap_values(
            instance,
            nsamples=self.nsamples,
        )

        attribution = self._extract_target_values(
            shap_values,
            target=target,
            n_features=len(
                self.feature_names
            ),
        )

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
            input_type="tabular",
            feature_names=list(
                self.feature_names
            ),
            metadata={
                "nsamples": self.nsamples,
                "background_size": int(
                    self._background_data.shape[0]
                ),
                "input_shape": list(
                    instance.shape
                ),
            },
        )

    @staticmethod
    def _extract_target_values(
        shap_values: Any,
        target: int,
        n_features: int,
    ) -> np.ndarray:
        """
        Normalize SHAP output formats across versions.
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

            # Some SHAP versions return:
            # [samples, features, classes]
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

        if len(values) != n_features:
            raise ValueError(
                "Number of SHAP values does not match "
                "number of features."
            )

        return values.astype(
            np.float32
        )