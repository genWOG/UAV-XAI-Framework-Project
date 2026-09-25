from dataclasses import dataclass
from typing import Any, Callable, Type

from .base import Explainer


@dataclass(frozen=True)
class ExplainerSpec:
    """
    Registration metadata for an XAI explainer.
    """

    explainer_cls: Type[Explainer]
    cache_instance: bool = False

    # Converts generic experiment context into kwargs
    # required by the explainer constructor.
    build_kwargs: Callable[
        [dict[str, Any]],
        dict[str, Any],
    ] | None = None

    build_input: Callable[
        [dict[str, Any]],
        Any,
    ] | None = None


class ExplainerRegistry:
    """
    Registry of XAI explainers.

    Explainers are resolved by:

        (input_type, method)
    """

    def __init__(self) -> None:
        self._explainers: dict[
            tuple[str, str],
            ExplainerSpec,
        ] = {}

    def register(
        self,
        input_type: str,
        method: str,
        explainer_cls: Type[Explainer],
        cache_instance: bool = False,
        build_kwargs: Callable[
            [dict[str, Any]],
            dict[str, Any],
        ] | None = None,
        build_input: Callable[
            [dict[str, Any]],
            Any,
        ] | None = None,
    ) -> None:
        input_type = input_type.strip().lower()
        method = method.strip().lower()

        if not input_type:
            raise ValueError(
                "input_type cannot be empty."
            )

        if not method:
            raise ValueError(
                "method cannot be empty."
            )

        if not issubclass(
            explainer_cls,
            Explainer,
        ):
            raise TypeError(
                "explainer_cls must inherit from Explainer."
            )

        key = (
            input_type,
            method,
        )

        if key in self._explainers:
            raise ValueError(
                f"Explainer for "
                f"input_type='{input_type}', "
                f"method='{method}' "
                f"is already registered."
            )

        self._explainers[key] = ExplainerSpec(
            explainer_cls=explainer_cls,
            cache_instance=cache_instance,
            build_kwargs=build_kwargs,
            build_input=build_input,
        )

    def build_input_data(
        self,
        input_type: str,
        method: str,
        context: dict[str, Any] | None = None,
        default_input: Any = None,
    ) -> Any:
        """
        Build the input representation expected by an explainer.

        If no input builder is registered, default_input is returned.
        """

        spec = self.get_spec(
            input_type=input_type,
            method=method,
        )

        if spec.build_input is None:
            return default_input

        return spec.build_input(
            context or {}
        )
    
    def get(
        self,
        input_type: str,
        method: str,
    ) -> Type[Explainer]:
        return self.get_spec(
            input_type,
            method,
        ).explainer_cls

    def get_spec(
        self,
        input_type: str,
        method: str,
    ) -> ExplainerSpec:
        input_type = input_type.strip().lower()
        method = method.strip().lower()

        key = (
            input_type,
            method,
        )

        try:
            return self._explainers[key]

        except KeyError as exc:
            available = ", ".join(
                f"{kind}:{name}"
                for kind, name in sorted(
                    self._explainers
                )
            )

            raise KeyError(
                f"Unknown explainer "
                f"'{input_type}:{method}'. "
                f"Available explainers: "
                f"{available or 'none'}"
            ) from exc

    def build_explainer_kwargs(
        self,
        input_type: str,
        method: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build constructor kwargs for a registered explainer.

        Explainer-specific configuration lives in the
        registration layer instead of the ExperimentRunner.
        """

        spec = self.get_spec(
            input_type,
            method,
        )

        if spec.build_kwargs is None:
            return {}

        return spec.build_kwargs(
            context or {}
        )

    def list_methods(
        self,
        input_type: str | None = None,
    ) -> list[str]:
        if input_type is None:
            return sorted(
                {
                    method
                    for _, method
                    in self._explainers
                }
            )

        input_type = input_type.strip().lower()

        return sorted(
            {
                method
                for kind, method
                in self._explainers
                if kind == input_type
            }
        )

    def list_entries(
        self,
    ) -> list[tuple[str, str]]:
        return sorted(
            self._explainers.keys()
        )

    def __contains__(
        self,
        key: tuple[str, str],
    ) -> bool:
        return key in self._explainers