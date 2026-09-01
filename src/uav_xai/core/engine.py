from typing import Any

from .base import Explainer
from .registry import ExplainerRegistry
from .result import ExplanationResult


class XAIEngine:
    """
    High-level interface for XAI methods.

    Explainer selection and constructor-specific configuration
    are delegated to the registry.
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
            tuple[str, str, int],
            Explainer,
        ] = {}

    def register(
        self,
        input_type: str,
        method: str,
        explainer_cls: type[Explainer],
        cache_instance: bool = False,
        build_kwargs: Any = None,
        build_input: Any = None,
    ) -> None:
        self.registry.register(
            input_type=input_type,
            method=method,
            explainer_cls=explainer_cls,
            cache_instance=cache_instance,
            build_kwargs=build_kwargs,
            build_input=build_input,
        )

    def list_methods(
        self,
        input_type: str | None = None,
    ) -> list[str]:
        return self.registry.list_methods(
            input_type=input_type,
        )

    def list_methods_all(
        self,
    ) -> list[tuple[str, str]]:
        return self.registry.list_entries()

    def _get_instance(
        self,
        input_type: str,
        method: str,
        context: dict[str, Any] | None = None,
    ) -> Explainer:

        context = (
            context
            if context is not None
            else {}
        )

        spec = self.registry.get_spec(
            input_type=input_type,
            method=method,
        )

        constructor_kwargs = (
            self.registry.build_explainer_kwargs(
                input_type=input_type,
                method=method,
                context=context,
            )
        )

        # Cached explainers may own resources,
        # for example PyTorch hooks.
        if not spec.cache_instance:

            return spec.explainer_cls(
                **constructor_kwargs
            )

        resource_key = id(
            context.get("target_layer")
        )

        cache_key = (
            input_type,
            method,
            resource_key,
        )

        if cache_key not in self._explainer_instances:

            self._explainer_instances[
                cache_key
            ] = spec.explainer_cls(
                **constructor_kwargs
            )

        return self._explainer_instances[
            cache_key
        ]

    def explain(
        self,
        input_type: str,
        method: str,
        model: Any,
        input_data: Any,
        target: int | str | None = None,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ExplanationResult:
        """
        Generate an explanation using a registered method.

        ``context`` contains shared experiment information,
        while additional keyword arguments remain supported
        for backward compatibility and are merged into the
        explainer context.
        """

        merged_context: dict[str, Any] = {}

        if context is not None:
            merged_context.update(context)

        merged_context.update(kwargs)

        merged_context.setdefault(
            "model",
            model,
        )

        merged_context.setdefault(
            "input_data",
            input_data,
        )

        merged_context.setdefault(
            "target",
            target,
        )

        explainer = self._get_instance(
            input_type=input_type,
            method=method,
            context=merged_context,
        )

        resolved_input = self.registry.build_input_data(
            input_type=input_type,
            method=method,
            context=merged_context,
            default_input=input_data,
        )

        return explainer.explain(
            model=model,
            input_data=resolved_input,
            target=target,
        )

    def close(self) -> None:
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