from abc import ABC, abstractmethod
from typing import Any

from .result import ExplanationResult


class Explainer(ABC):
    """
    Base interface for all XAI methods.
    """

    @abstractmethod
    def explain(
        self,
        model: Any,
        input_data: Any,
        target: int | str | None = None,
    ) -> ExplanationResult:
        """
        Generate an explanation for a model prediction.

        Parameters
        ----------
        model:
            Model to explain.
        input_data:
            Input passed to the model.
        target:
            Target class/action.

        Returns
        -------
        ExplanationResult
            Standardized explanation result.
        """
        raise NotImplementedError