from typing import Any

from .base import Explainer
from .registry import ExplainerRegistry
from .result import ExplanationResult


class XAIEngine:
    """
    High-level interface for XAI methods.

    The engine manages explainer registration and creates
    explainer instances on demand.
    """

    def __init__(
        self,
        registry: ExplainerRegistry | None = None,
    ) -> None:
        self.registry = (
            registry
            if registry is not None
            else ExplainerRegistry()
        )

        self._explainer_instances: dict[
            tuple[str, int],
            Explainer,
        ] = {}

    def register(
        self,
        name: str,
        explainer_cls: type[Explainer],
    ) -> None:
        """
        Register an XAI method.
        """

        self.registry.register(
            name,
            explainer_cls,
        )

    def list_methods(self) -> list[str]:
        """
        Return available XAI methods.
        """

        return self.registry.list_methods()

    def _get_instance(
        self,
        method: str,
        target_layer: Any = None,
    ) -> Explainer:
        """
        Create or reuse an explainer instance.

        For Grad-CAM, target_layer is part of the instance
        configuration.
        """

        explainer_cls = self.registry.get(
            method
        )

        cache_key = (
            method,
            id(target_layer),
        )

        if cache_key not in self._explainer_instances:

            if method == "grad_cam":
                if target_layer is None:
                    raise ValueError(
                        "target_layer is required "
                        "for Grad-CAM."
                    )

                explainer = explainer_cls(
                    target_layer=target_layer
                )

            else:
                explainer = explainer_cls()

            self._explainer_instances[
                cache_key
            ] = explainer

        return self._explainer_instances[
            cache_key
        ]

    def explain(
        self,
        method: str,
        model: Any,
        input_data: Any,
        target: int | str | None = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        """
        Generate an explanation using a registered method.

        Parameters
        ----------
        method:
            Name of the XAI method.

        model:
            Model being explained.

        input_data:
            Model input.

        target:
            Target class/action.

        kwargs:
            Method-specific configuration.

        Returns
        -------
        ExplanationResult
        """

        target_layer = kwargs.pop(
            "target_layer",
            None,
        )

        explainer = self._get_instance(
            method=method,
            target_layer=target_layer,
        )

        return explainer.explain(
            model=model,
            input_data=input_data,
            target=target,
        )

    def close(self) -> None:
        """
        Release explainer resources and hooks.
        """

        for explainer in (
            self._explainer_instances.values()
        ):
            close_method = getattr(
                explainer,
                "close",
                None,
            )

            if close_method is not None:
                close_method()

        self._explainer_instances.clear()

    def __enter__(self) -> "XAIEngine":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()