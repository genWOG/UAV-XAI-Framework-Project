import inspect
from typing import Any

from .base import Explainer
from .registry import ExplainerRegistry
from .result import ExplanationResult
from .target import XAITarget


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
        # --- ФИКС ДЛЯ GRAD-CAM ---
        if context and "target_layer" in context and context["target_layer"] is not None:
            constructor_kwargs["target_layer"] = context["target_layer"]
        # --------------------------

        # --- ФИКС ДЛЯ LIME/SHAP: preprocess ---
        if context and "preprocess" in context and context["preprocess"] is not None:
            constructor_kwargs["preprocess"] = context["preprocess"]
        # ---------------------------------------
        
        # --- ФИКС: фильтруем kwargs, которые explainer не принимает ---
        init_signature = inspect.signature(spec.explainer_cls.__init__)
        accepts_var_kwargs_init = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in init_signature.parameters.values()
        )
        if not accepts_var_kwargs_init:
            constructor_kwargs = {
                k: v
                for k, v in constructor_kwargs.items()
                if k in init_signature.parameters
            }
        # --------------------------

        if not spec.cache_instance:
            return spec.explainer_cls(**constructor_kwargs)

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

        if (
            cache_key
            not in self._explainer_instances
        ):
            self._explainer_instances[
                cache_key
            ] = spec.explainer_cls(
                **constructor_kwargs
            )

        return self._explainer_instances[
            cache_key
        ]

    def _resolve_method_target(
        self,
        target: int | str | XAITarget | None,
    ) -> int | str | XAITarget | None:
        """
        Convert classification XAITarget into the primitive
        class-id representation expected by legacy classification
        explainers.

        Detection targets remain structured XAITarget objects.
        """

        if isinstance(
            target,
            XAITarget,
        ):
            if target.task == "classification":
                return target.class_id

        return target

    def explain(
        self,
        input_type: str,
        method: str,
        model: Any,
        input_data: Any,
        target: Any,
        context: dict[str, Any] | None = None,
        **kwargs,   # <--- ДОБАВИТЬ ВОТ ЭТО
    ) -> ExplanationResult:
        """
        Generate an explanation using a registered method.

        `context` contains shared experiment information.

        Additional keyword arguments are merged into context
        for backward compatibility.
        """

        merged_context: dict[str, Any] = {}

        if context is not None:
            merged_context.update(
                context
            )

        merged_context.update(
            kwargs
        )

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

        # --------------------------------------------------------
        # Resolve explainer.
        # --------------------------------------------------------

        explainer = self._get_instance(
            input_type=input_type,
            method=method,
            context=merged_context,
        )

        # --------------------------------------------------------
        # Resolve explainer-specific input.
        # --------------------------------------------------------

        resolved_input = (
            self.registry.build_input_data(
                input_type=input_type,
                method=method,
                context=merged_context,
                default_input=input_data,
            )
        )

        # --------------------------------------------------------
        # Adapt universal target to the target format expected
        # by the selected explainer.
        # --------------------------------------------------------

        method_target = (
            self._resolve_method_target(
                target
            )
        )

        # --------------------------------------------------------
        # Preserve compatibility with explainers that don't
        # declare a `context` parameter.
        # --------------------------------------------------------

        explain_kwargs: dict[
            str,
            Any,
        ] = {
            "model": model,
            "input_data": resolved_input,
            "target": method_target,
            "training_data": kwargs.get("training_data"),
            "feature_names": kwargs.get("feature_names"),
            "background_data": kwargs.get("background_data"),
        }

        signature = inspect.signature(
            explainer.explain
        )

        if "context" in signature.parameters:
            explain_kwargs[
                "context"
            ] = merged_context

        # --- ФИКС: Отфильтровываем kwargs, которые explainer не принимает ---
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not accepts_var_kwargs:
            explain_kwargs = {
                k: v
                for k, v in explain_kwargs.items()
                if k in signature.parameters
            }
        # ------------------------------------------------------------------

        return explainer.explain(
            **explain_kwargs
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