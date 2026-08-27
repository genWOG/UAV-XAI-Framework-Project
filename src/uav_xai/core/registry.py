from typing import Type

from .base import Explainer


class ExplainerRegistry:
    """
    Registry of available XAI explainers.

    The registry maps a method name such as ``"grad_cam"``
    to an explainer class.
    """

    def __init__(self) -> None:
        self._explainers: dict[str, Type[Explainer]] = {}

    def register(
        self,
        name: str,
        explainer_cls: Type[Explainer],
    ) -> None:
        """
        Register a new explainer class.

        Parameters
        ----------
        name:
            Public method name.

        explainer_cls:
            Explainer class.
        """

        if not name:
            raise ValueError(
                "Explainer name cannot be empty."
            )

        if not issubclass(
            explainer_cls,
            Explainer,
        ):
            raise TypeError(
                "explainer_cls must inherit from Explainer."
            )

        if name in self._explainers:
            raise ValueError(
                f"Explainer '{name}' is already registered."
            )

        self._explainers[name] = explainer_cls

    def get(
        self,
        name: str,
    ) -> Type[Explainer]:
        """
        Get explainer class by name.
        """

        try:
            return self._explainers[name]
        except KeyError as exc:
            available = ", ".join(
                sorted(self._explainers)
            )

            raise KeyError(
                f"Unknown explainer '{name}'. "
                f"Available explainers: {available or 'none'}"
            ) from exc

    def list_methods(self) -> list[str]:
        """
        Return all registered method names.
        """

        return sorted(
            self._explainers.keys()
        )

    def __contains__(
        self,
        name: str,
    ) -> bool:
        return name in self._explainers