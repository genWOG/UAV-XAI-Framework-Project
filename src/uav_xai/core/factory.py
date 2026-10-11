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

    target_layers = context.get(
        "target_layers"
    )

    if target_layers:
        return {
            "target_layer": target_layers,
        }

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
    """
    Resolve Grad-CAM input.

    Prefer explicit xai_input when a runner provides
    a model-specific XAI representation. Otherwise fall
    back to the generic input_data.
    """

    xai_input = context.get(
        "xai_input"
    )

    if xai_input is not None:
        return xai_input

    input_data = context.get(
        "input_data"
    )

    if input_data is not None:
        return input_data

    raise ValueError(
        "Grad-CAM requires xai_input or input_data."
    )


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
    preprocess = context.get("preprocess")
    xai_config = context.get("xai_config", {})
    method_configs = xai_config.get("method_configs", {})
    method_config = method_configs.get("lime", {}) if isinstance(method_configs, dict) else {}

    def value(name: str, default: Any) -> Any:
        return method_config.get(name, xai_config.get(name, default))

    kwargs = {
        "preprocess": preprocess,
        "num_samples": int(value("num_samples", 500)),
        "num_segments": int(value("num_segments", 100)),
        "compactness": float(value("compactness", 10.0)),
        "hide_color": value("hide_color", 0),
    }
    if context.get("task") == "detection":
        kwargs["target_iou_threshold"] = float(
            value("target_iou_threshold", xai_config.get("lime_iou_threshold", 0.5))
        )
    elif preprocess is None:
        raise ValueError("LIME Image requires preprocess.")
    return kwargs


def _shap_image_kwargs(
    context: dict[str, Any],
) -> dict[str, Any]:
    preprocess = context.get("preprocess")
    xai_config = context.get("xai_config", {})
    method_configs = xai_config.get("method_configs", {})
    method_config = method_configs.get("shap", {}) if isinstance(method_configs, dict) else {}

    def value(name: str, default: Any) -> Any:
        return method_config.get(name, xai_config.get(name, default))

    kwargs = {
        "preprocess": preprocess,
        "nsamples": int(value("nsamples", 100)),
        "num_segments": int(value("num_segments", 30)),
        "compactness": float(value("compactness", 10.0)),
        "hide_color": value("hide_color", 0),
        "baseline_mode": str(value("baseline_mode", "constant")),
    }
    if context.get("task") == "detection":
        kwargs["target_iou_threshold"] = float(
            value(
                "target_iou_threshold",
                xai_config.get("shap_iou_threshold", xai_config.get("lime_iou_threshold", 0.5)),
            )
        )
    elif preprocess is None:
        raise ValueError("SHAP Image requires preprocess.")
    return kwargs


def _lrp_image_kwargs(
    context: dict[str, Any],
) -> dict[str, Any]:
    preprocess = context.get("preprocess")
    if context.get("task") != "detection" and preprocess is None:
        raise ValueError("LRP Image requires preprocess.")

    xai_config = context.get("xai_config", {})
    method_configs = xai_config.get("method_configs", {})
    method_config = method_configs.get("lrp", {}) if isinstance(method_configs, dict) else {}

    def value(name: str, default: Any) -> Any:
        return method_config.get(name, xai_config.get(name, default))

    kwargs = {
        "preprocess": preprocess,
        "epsilon": float(value("epsilon", 1e-6)),
        "gamma": float(value("gamma", 0.1)),
        "low": float(value("low", 0.0)),
        "high": float(value("high", 1.0)),
    }
    if context.get("task") == "detection":
        kwargs["context"] = context
    return kwargs


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