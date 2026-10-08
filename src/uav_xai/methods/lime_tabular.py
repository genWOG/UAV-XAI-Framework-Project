import time
from typing import Any, Callable

import numpy as np
from lime.lime_tabular import LimeTabularExplainer as LimeLibraryExplainer

from ..core.base import Explainer
from ..core.result import ExplanationResult


class LimeTabularExplainer(Explainer):
    """
    LIME explainer for tabular data.

    Works with black-box classifiers exposing either:

    - predict_proba(X)

    or a callable prediction function returning
    class probabilities.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: list[str],
        class_names: list[str] | None = None,
        num_samples: int = 1000,
        discretize_continuous: bool = True,
        random_state: int | None = 42,
    ) -> None:

        training_data = np.asarray(
            training_data,
            dtype=np.float32,
        )

        if training_data.ndim != 2:
            raise ValueError(
                "training_data must be a 2D array."
            )

        if training_data.shape[1] != len(
            feature_names
        ):
            raise ValueError(
                "Number of feature_names must match "
                "training_data columns."
            )

        if num_samples < 1:
            raise ValueError(
                "num_samples must be >= 1."
            )

        self.feature_names = feature_names
        self.class_names = class_names
        self.num_samples = num_samples

        self._explainer = LimeLibraryExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification",
            discretize_continuous=discretize_continuous,
            random_state=random_state,
        )

    def _get_predict_fn(
        self,
        model: Any,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """
        Resolve a prediction function from the model.
        """

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

        if instance.ndim == 2:
            if instance.shape[0] != 1:
                raise ValueError(
                    "Only one sample can be explained at a time."
                )

            instance = instance[0]

        if instance.ndim != 1:
            raise ValueError(
                "input_data must have shape [features] "
                "or [1, features]."
            )

        if len(instance) != len(
            self.feature_names
        ):
            raise ValueError(
                "Number of input features must match "
                "feature_names."
            )

        predict_fn = self._get_predict_fn(
            model
        )

        # If target is not explicitly provided,
        # use the model's predicted class.
        if target is None:

            probabilities = predict_fn(
                instance.reshape(1, -1)
            )

            target = int(
                np.argmax(
                    probabilities[0]
                )
            )

        explanation = self._explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=len(
                self.feature_names
            ),
            num_samples=self.num_samples,
            labels=[int(target)],
        )

        # Dense attribution vector.
        attribution = np.zeros(
            len(self.feature_names),
            dtype=np.float32,
        )

        for feature_idx, weight in (
            explanation.as_map()[
                int(target)
            ]
        ):
            attribution[
                feature_idx
            ] = weight

        probabilities = predict_fn(
            instance.reshape(1, -1)
        )

        confidence = float(
            probabilities[
                0,
                int(target),
            ]
        )

        runtime_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        return ExplanationResult(
            method="lime",
            attribution=attribution,
            target=int(target),
            confidence=confidence,
            runtime_ms=runtime_ms,
            input_type="tabular",
            feature_names=list(
                self.feature_names
            ),
            metadata={
                "num_samples": self.num_samples,
                "class_names": self.class_names,
                "input_shape": list(instance.shape),
            },
        )