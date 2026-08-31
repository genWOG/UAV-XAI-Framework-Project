import numpy as np

from typing import Any

from .engine import XAIEngine

from ..methods import (
    GradCAMExplainer,
    LimeImageExplainer,
    LimeTabularExplainer,
    ShapTabularExplainer,
    ShapImageExplainer,
    LrpImageExplainer,
)


def _grad_cam_kwargs(
    context: dict[str, Any],
) -> dict[str, Any]:
    target_layer = context.get(
        "target_layer"
    )

    if target_layer is None:
        raise ValueError(
            "Grad-CAM requires target_layer."
        )

    return {
        "target_layer": target_layer,
    }

def _grad_cam_input(
    context: dict[str, Any],
) -> Any:
    input_tensor = context.get(
        "input_tensor"
    )

    if input_tensor is None:
        raise ValueError(
            "Grad-CAM requires input_tensor."
        )

    return input_tensor


def _lime_image_input(
    context: dict[str, Any],
) -> Any:
    image = context.get(
        "original_image"
    )

    if image is None:
        raise ValueError(
            "LIME Image requires original_image."
        )

    return np.asarray(
        image
    )

def _lime_image_kwargs(
    context: dict[str, Any],
) -> dict[str, Any]:
    preprocess = context.get(
        "preprocess"
    )

    if preprocess is None:
        raise ValueError(
            "LIME Image requires preprocess."
        )

    xai_config = context.get(
        "xai_config",
        {},
    )

    return {
        "preprocess": preprocess,
        "num_samples": int(
            xai_config.get(
                "num_samples",
                500,
            )
        ),
    }

def _shap_image_kwargs(
    context: dict[str, Any],
) -> dict[str, Any]:
    preprocess = context.get(
        "preprocess"
    )

    if preprocess is None:
        raise ValueError(
            "SHAP Image requires preprocess."
        )

    xai_config = context.get(
        "xai_config",
        {},
    )

    return {
        "preprocess": preprocess,
        "nsamples": int(
            xai_config.get(
                "nsamples",
                100,
            )
        ),
        "num_segments": int(
            xai_config.get(
                "num_segments",
                30,
            )
        ),
        "compactness": float(
            xai_config.get(
                "compactness",
                10.0,
            )
        ),
        "hide_color": xai_config.get(
            "hide_color",
            0,
        ),
    }

def _lrp_image_kwargs(
    context: dict[str, Any],
) -> dict[str, Any]:
    preprocess = context.get(
        "preprocess"
    )

    if preprocess is None:
        raise ValueError(
            "LRP Image requires preprocess."
        )

    xai_config = context.get(
        "xai_config",
        {},
    )

    return {
        "preprocess": preprocess,
        "epsilon": float(
            xai_config.get(
                "epsilon",
                1e-6,
            )
        ),
        "gamma": float(
            xai_config.get(
                "gamma",
                0.1,
            )
        ),
        "low": float(
            xai_config.get(
                "low",
                0.0,
            )
        ),
        "high": float(
            xai_config.get(
                "high",
                1.0,
            )
        ),
    }

def _lime_tabular_kwargs(
    context: dict[str, Any],
) -> dict[str, Any]:
    required = (
        "training_data",
        "feature_names",
    )

    missing = [
        name
        for name in required
        if name not in context
    ]

    if missing:
        raise ValueError(
            "LIME Tabular requires: "
            + ", ".join(missing)
        )

    return {
        "training_data": context["training_data"],
        "feature_names": context["feature_names"],
        "class_names": context.get(
            "class_names"
        ),
        "num_samples": int(
            context.get(
                "num_samples",
                1000,
            )
        ),
    }


def _shap_tabular_kwargs(
    context: dict[str, Any],
) -> dict[str, Any]:
    required = (
        "background_data",
        "feature_names",
    )

    missing = [
        name
        for name in required
        if name not in context
    ]

    if missing:
        raise ValueError(
            "SHAP Tabular requires: "
            + ", ".join(missing)
        )

    return {
        "background_data": context[
            "background_data"
        ],
        "feature_names": context[
            "feature_names"
        ],
        "nsamples": int(
            context.get(
                "nsamples",
                100,
            )
        ),
    }


def create_default_engine() -> XAIEngine:
    engine = XAIEngine()

    # ------------------------------
    # Image
    # ------------------------------

    engine.register(
        input_type="image",
        method="grad_cam",
        explainer_cls=GradCAMExplainer,
        cache_instance=True,
        build_kwargs=_grad_cam_kwargs,
        build_input=_grad_cam_input,
    )

    engine.register(
        input_type="image",
        method="lime",
        explainer_cls=LimeImageExplainer,
        build_kwargs=_lime_image_kwargs,
        build_input=_lime_image_input,
    )

    engine.register(
        input_type="image",
        method="shap",
        explainer_cls=ShapImageExplainer,
        build_kwargs=_shap_image_kwargs,
        build_input=_lime_image_input,
    )

    engine.register(
        input_type="image",
        method="lrp",
        explainer_cls=LrpImageExplainer,
        build_kwargs=_lrp_image_kwargs,
        build_input=_lime_image_input,
    )

    # ------------------------------
    # Tabular
    # ------------------------------

    engine.register(
        input_type="tabular",
        method="lime",
        explainer_cls=LimeTabularExplainer,
        build_kwargs=_lime_tabular_kwargs,
    )

    engine.register(
        input_type="tabular",
        method="shap",
        explainer_cls=ShapTabularExplainer,
        build_kwargs=_shap_tabular_kwargs,
    )

    return engine