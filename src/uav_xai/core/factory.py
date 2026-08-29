from .engine import XAIEngine
from ..methods import (
    GradCAMExplainer,
    LimeImageExplainer,
    LimeTabularExplainer,
    ShapTabularExplainer,
)


def create_default_engine() -> XAIEngine:
    """
    Create an XAI engine with all currently available
    built-in explainers.
    """

    engine = XAIEngine()

    engine.register(
        "grad_cam",
        GradCAMExplainer,
    )

    engine.register(
        "lime_image",
        LimeImageExplainer,
    )

    engine.register(
        "lime_tabular",
        LimeTabularExplainer,
    )

    engine.register(
        "shap_tabular",
        ShapTabularExplainer,
    )

    return engine