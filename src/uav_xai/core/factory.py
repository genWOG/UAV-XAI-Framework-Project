from .engine import XAIEngine

# Image explainers
from ..methods.gradcam import GradCAMExplainer
from ..methods.lime_image import LimeImageExplainer
from ..methods.shap_image import ShapImageExplainer
from ..methods.lrp_image import LrpImageExplainer

# Tabular explainers
from ..methods.lime_tabular import LimeTabularExplainer
from ..methods.shap_tabular import ShapTabularExplainer
from ..methods.counterfactual_tabular import CounterfactualTabularExplainer

import numpy as np


def _build_image_input_for_perturbation(context: dict) -> np.ndarray:
    """
    LIME/SHAP работают на оригинальной RGB-картинке [H, W, 3].
    Берём её из context["original_image"] (PIL.Image).
    Если её нет — используем input_data, но тогда нужен numpy.
    """
    original = context.get("original_image")
    if original is not None:
        # PIL.Image -> numpy [H, W, 3] uint8
        return np.asarray(original.convert("RGB"))

    fallback = context.get("input_data")
    if fallback is None:
        raise ValueError("No image available for LIME/SHAP input.")
    if hasattr(fallback, "detach"):
        fallback = fallback.detach().cpu()
        if fallback.ndim == 4:
            fallback = fallback[0]
        if fallback.shape[0] == 3:
            fallback = fallback.permute(1, 2, 0)
        fallback = fallback.numpy()
        if fallback.max() <= 1.0 + 1e-6:
            fallback = (fallback * 255.0).astype(np.uint8)
    return np.asarray(fallback)


def create_default_engine() -> XAIEngine:
    engine = XAIEngine()

    # --- Image methods ---
    engine.register("image", "grad_cam", GradCAMExplainer)
    engine.register(
        "image", "lime", LimeImageExplainer,
        build_input=_build_image_input_for_perturbation,
    )
    engine.register(
        "image", "shap", ShapImageExplainer,
        build_input=_build_image_input_for_perturbation,
    )
    engine.register("image", "lrp", LrpImageExplainer)

    # --- Tabular methods ---
    engine.register("tabular", "lime", LimeTabularExplainer)
    engine.register("tabular", "shap", ShapTabularExplainer)
    engine.register("tabular", "counterfactual", CounterfactualTabularExplainer)

    return engine