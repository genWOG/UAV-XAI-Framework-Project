from .engine import XAIEngine
from ..methods import GradCAMExplainer


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

    return engine