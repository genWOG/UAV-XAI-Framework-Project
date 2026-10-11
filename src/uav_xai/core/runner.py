import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)

from .config import load_config
from .predictor import ModelPredictor
from .target import XAITarget
from .target_resolver import TargetResolver

from ..metrics import (
    compute_deletion,
    compute_insertion,
    compute_detection_deletion,
    compute_detection_insertion,
)

from ..utils import (
    overlay_heatmap,
    plot_faithfulness_curves,
    save_image,
)

from ..core import create_default_engine


def _prepare_yolo_xai_input(
    image: Image.Image,
    size: int = 640,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
]:
    """
    Prepare an autograd-compatible YOLO tensor.

    Uses standard square letterbox preprocessing:
        resize while preserving aspect ratio;
        pad with value 114;
        convert RGB image into [1, 3, H, W];
        normalize to [0, 1].

    Returns
    -------
    tensor:
        [1, 3, size, size]

    metadata:
        Letterbox transform parameters.
    """

    image_np = np.asarray(
        image.convert("RGB")
    )

    original_height, original_width = (
        image_np.shape[:2]
    )

    scale = min(
        size / original_height,
        size / original_width,
    )

    resized_width = int(
        round(
            original_width * scale
        )
    )

    resized_height = int(
        round(
            original_height * scale
        )
    )

    resized = cv2.resize(
        image_np,
        (
            resized_width,
            resized_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (
            size,
            size,
            3,
        ),
        114,
        dtype=np.uint8,
    )

    pad_x = (
        size - resized_width
    ) // 2

    pad_y = (
        size - resized_height
    ) // 2

    canvas[
        pad_y:
        pad_y + resized_height,
        pad_x:
        pad_x + resized_width,
    ] = resized

    tensor = (
        torch.from_numpy(
            canvas
        )
        .permute(
            2,
            0,
            1,
        )
        .contiguous()
        .float()
        / 255.0
    )

    tensor = tensor.unsqueeze(0)

    metadata = {
        "scale": float(scale),
        "pad_x": int(pad_x),
        "pad_y": int(pad_y),
        "original_width": int(
            original_width
        ),
        "original_height": int(
            original_height
        ),
        "input_width": int(size),
        "input_height": int(size),
        "resized_width": int(
            resized_width
        ),
        "resized_height": int(
            resized_height
        ),
    }

    return tensor, metadata


def _resolve_module(
    root: torch.nn.Module,
    path: str,
) -> torch.nn.Module:
    """
    Resolve a module from a path.

    Supported examples:

        layer4[-1].conv2
        16
        model.16
    """

    current: Any = root

    for part in path.split("."):

        if part.isdigit():

            index = int(part)

            if hasattr(
                current,
                "model",
            ):
                current = current.model[
                    index
                ]

            elif isinstance(
                current,
                (list, tuple),
            ):
                current = current[
                    index
                ]

            else:
                raise ValueError(
                    f"Cannot resolve numeric "
                    f"layer '{part}' from "
                    f"{type(current).__name__}."
                )

            continue

        if (
            "["
            in part
            and part.endswith("]")
        ):

            name, index_text = (
                part.split(
                    "[",
                    1,
                )
            )

            index = int(
                index_text[:-1]
            )

            current = getattr(
                current,
                name,
            )

            current = current[
                index
            ]

        else:

            current = getattr(
                current,
                part,
            )

    if not isinstance(
        current,
        torch.nn.Module,
    ):
        raise TypeError(
            f"Resolved object '{path}' "
            "is not a torch.nn.Module."
        )

    return current


def _json_default(
    value: Any,
) -> Any:

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        torch.Tensor,
    ):
        return (
            value.detach()
            .cpu()
            .tolist()
        )

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    raise TypeError(
        f"Object of type "
        f"{type(value).__name__} "
        "is not JSON serializable."
    )


class ExperimentRunner:
    """
    Run a complete configurable XAI experiment.

    Supports:
        - classification;
        - object detection;
        - multiple XAI methods;
        - structured detection targets;
        - multi-scale YOLO Grad-CAM;
        - per-method artifacts;
        - classification faithfulness metrics.
    """

    def __init__(
        self,
        config_path: str | Path,
    ) -> None:

        self.config_path = Path(
            config_path
        )

        self.config = load_config(
            self.config_path
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.output_dir: (
            Path | None
        ) = None

    # ============================================================
    # Output
    # ============================================================

    def _create_output_dir(
        self,
    ) -> Path:

        name = self.config[
            "experiment"
        ][
            "name"
        ]

        timestamp = (
            datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        root = Path(
            self.config[
                "output"
            ][
                "root"
            ]
        )

        output_dir = root / (
            f"{name}_{timestamp}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        self.output_dir = output_dir

        return output_dir

    def _save_config(
        self,
        output_dir: Path,
    ) -> None:

        shutil.copy2(
            self.config_path,
            output_dir
            / "config.yaml",
        )

    # ============================================================
    # Model
    # ============================================================

    def _load_model(
        self,
    ) -> dict[str, Any]:

        model_config = self.config[
            "model"
        ]

        model_name = (
            str(
                model_config[
                    "name"
                ]
            )
            .strip()
            .lower()
        )

        task = (
            str(
                model_config.get(
                    "task",
                    "classification",
                )
            )
            .strip()
            .lower()
        )

        if model_name == "resnet18":

            if task != "classification":
                raise ValueError(
                    "ResNet-18 supports "
                    "classification only."
                )

            weights = (
                ResNet18_Weights.DEFAULT
            )

            model = resnet18(
                weights=weights
            )

            model = model.to(
                self.device
            )

            model.eval()

            return {
                "model": model,
                "weights": weights,
                "task": "classification",
                "name": model_name,
            }

        if model_name.startswith(
            "yolo"
        ):

            from ultralytics import YOLO

            model_path = (
                model_config.get(
                    "weights",
                    f"{model_name}.pt",
                )
            )

            model = YOLO(
                model_path
            )

            model.model.eval()

            return {
                "model": model,
                "weights": None,
                "task": task,
                "name": model_name,
            }

        raise ValueError(
            f"Unsupported model: "
            f"{model_name}"
        )

    # ============================================================
    # Image
    # ============================================================

    def _load_image(
        self,
        weights: Any,
        task: str,
    ) -> tuple[
        Image.Image,
        Any,
    ]:

        image_path = Path(
            self.config[
                "image"
            ][
                "path"
            ]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        if task == "classification":

            if weights is None:
                raise ValueError(
                    "Classification model "
                    "requires weights."
                )

            transform = (
                weights.transforms()
            )

            input_data = (
                transform(
                    image
                )
                .unsqueeze(0)
                .to(self.device)
            )

            return (
                image,
                input_data,
            )

        if task == "detection":

            return (
                image,
                str(image_path),
            )

        raise ValueError(
            f"Unsupported task: "
            f"{task}"
        )

    # ============================================================
    # Methods
    # ============================================================

    def _get_methods(
        self,
    ) -> list[str]:

        xai_config = self.config[
            "xai"
        ]

        if "methods" in xai_config:

            methods = xai_config[
                "methods"
            ]

            if not isinstance(
                methods,
                list,
            ):
                raise ValueError(
                    "xai.methods must "
                    "be a list."
                )

            if not methods:
                raise ValueError(
                    "xai.methods cannot "
                    "be empty."
                )

            return [
                str(method)
                .strip()
                .lower()
                for method in methods
            ]

        if "method" in xai_config:

            return [
                str(
                    xai_config[
                        "method"
                    ]
                )
                .strip()
                .lower()
            ]

        raise ValueError(
            "XAI configuration must contain "
            "'method' or 'methods'."
        )

    # ============================================================
    # Target layer
    # ============================================================

    def _resolve_xai_target_layer(
        self,
        model: Any,
        task: str,
        target_layer_path: (
            str | None
        ),
    ) -> Any:

        if target_layer_path:

            root = (
                model.model
                if task == "detection"
                else model
            )

            return _resolve_module(
                root,
                target_layer_path,
            )

        if task == "classification":
            return None

        if task != "detection":
            return None

        network = getattr(
            model,
            "model",
            None,
        )

        if network is None:
            raise ValueError(
                "Detection model does not expose "
                "its internal model."
            )

        layers = getattr(
            network,
            "model",
            None,
        )

        if layers is None:
            raise ValueError(
                "Detection model does not expose "
                "its layer list."
            )

        # YOLO26 Detect receives:
        #
        #   [16, 19, 22]
        #
        # corresponding to P3/P4/P5.
        #
        # Use all three scales for detection XAI.
        target_indices = [
            16,
            19,
            22,
        ]

        target_layers = []

        for index in target_indices:

            if index >= len(layers):
                raise ValueError(
                    f"Target layer {index} "
                    f"is unavailable; "
                    f"model contains "
                    f"{len(layers)} layers."
                )

            layer = layers[
                index
            ]

            if not isinstance(
                layer,
                torch.nn.Module,
            ):
                raise TypeError(
                    f"Target layer {index} "
                    "is not a torch.nn.Module."
                )

            target_layers.append(
                layer
            )

        return target_layers

    # ============================================================
    # Context
    # ============================================================

    def _build_xai_context(
        self,
        model: Any,
        weights: Any,
        original_image: Image.Image,
        input_data: Any,
        xai_input: torch.Tensor,
        xai_target: Any,
        task: str,
        letterbox: (
            dict[str, Any]
            | None
        ),
    ) -> dict[str, Any]:

        xai_config = self.config[
            "xai"
        ]

        target_layer = (
            self._resolve_xai_target_layer(
                model=model,
                task=task,
                target_layer_path=(
                    xai_config.get(
                        "target_layer"
                    )
                ),
            )
        )

        if isinstance(
            target_layer,
            (list, tuple),
        ):

            target_layers = list(
                target_layer
            )

            single_target_layer = None

        else:

            target_layers = None
            single_target_layer = (
                target_layer
            )

        preprocess = None

        if (
            task == "classification"
            and weights is not None
        ):

            transform = (
                weights.transforms()
            )

            def preprocess(
                rgb_image: np.ndarray,
            ) -> torch.Tensor:

                pil_image = (
                    Image.fromarray(
                        rgb_image.astype(
                            np.uint8
                        )
                    )
                )

                return transform(
                    pil_image
                )

        return {
            "model": model,
            "weights": weights,
            "task": task,
            "input_type": xai_config.get(
                "input_type",
                "image",
            ),
            "xai_config": xai_config,
            "target_layer": single_target_layer,
            "target_layers": target_layers,
            "preprocess": preprocess,
            "input_data": input_data,
            "xai_input": xai_input,
            "xai_target": xai_target,
            "original_image": original_image,
            "letterbox": letterbox,

            "detection_iou_threshold": float(
                xai_config.get(
                    "detection_iou_threshold",
                    0.10,
                )
            ),

            "detection_score_threshold": float(
                xai_config.get(
                    "detection_score_threshold",
                    0.01,
                )
            ),

            "detection_top_k": int(
                xai_config.get(
                    "detection_top_k",
                    16,
                )
            ),
        }

    # ============================================================
    # One explanation
    # ============================================================

    def _run_explanation(
        self,
        engine: Any,
        model: Any,
        prediction: Any,
        original_image: Image.Image,
        input_data: Any,
        context: dict[str, Any],
        method: str,
    ) -> dict[str, Any]:

        xai_target = context["xai_target"]

        if xai_target.task == "classification":
            method_target = xai_target.class_id
        else:
            method_target = xai_target

        explanation = engine.explain(
            input_type=context["input_type"],
            method=method,
            model=model,
            input_data=context["xai_input"],
            target=method_target,
            context=context,
        )

        # --------------------------------------------------------
        # Visualization
        #
        # IMPORTANT:
        # do not bbox-mask detection CAM here.
        # The raw multi-query/multi-scale attribution is what
        # we want to inspect first.
        # --------------------------------------------------------

        visualization_path = (
            self.output_dir
            / f"{method}.png"
        )

        # Keep CAM in model-input coordinates for metrics.
        # Preserve the raw attribution for metric preprocessing.
        metric_attribution = np.asarray(
            explanation.attribution,
            dtype=np.float32,
        ).copy()

        heatmap = metric_attribution.copy()

        if context["task"] == "detection":
            letterbox = context["letterbox"]

            model_shape = (
                int(letterbox["input_height"]),
                int(letterbox["input_width"]),
            )
            original_shape = (
                int(letterbox["original_height"]),
                int(letterbox["original_width"]),
            )

            if heatmap.shape == model_shape:
                # Grad-CAM: model-input coordinates -> original image.
                heatmap = self._restore_detection_heatmap(
                    heatmap=heatmap,
                    letterbox=letterbox,
                    output_size=original_image.size,
                )
            elif heatmap.shape == original_shape:
                # LIME/SHAP/LRP image maps are already in original-image
                # coordinates. Do not letterbox them a second time.
                pass
            else:
                raise ValueError(
                    "Unexpected detection attribution shape "
                    f"{heatmap.shape}; expected model shape "
                    f"{model_shape} or original image shape "
                    f"{original_shape}."
                )

            explanation.attribution = heatmap

        visualization = (
            overlay_heatmap(
                image=original_image,
                heatmap=heatmap,
                alpha=0.45,
            )
        )

        save_image(
            visualization,
            visualization_path,
        )

        # --------------------------------------------------------
        # Metrics
        # --------------------------------------------------------

        metrics_config = (
            self.config.get(
                "metrics",
                {},
            )
        )

        metrics_enabled = (
            bool(
                metrics_config.get(
                    "enabled",
                    True,
                )
            )
            and context["task"] in {
                "classification",
                "detection",
            }
        )

        deletion_result = None
        insertion_result = None
        metrics_path = None

        if metrics_enabled:

            steps = int(
                metrics_config.get(
                    "steps",
                    20,
                )
            )

            baseline_value = float(
                metrics_config.get(
                    "baseline_value",
                    0.0,
                )
            )

            attribution = metric_attribution
            metric_iou_threshold = None

            if context["task"] == "detection":

                metric_predictor = ModelPredictor(
                    model=model,
                    task="detection",
                )

                metric_input_tensor = (
                    context["xai_input"].detach()
                )

                letterbox = context["letterbox"]

                metric_shape = tuple(
                    metric_input_tensor.shape[-2:]
                )

                original_shape = (
                    int(letterbox["original_height"]),
                    int(letterbox["original_width"]),
                )

                if metric_attribution.shape == original_shape:
                    resized_attribution = cv2.resize(
                        metric_attribution,
                        (
                            int(letterbox["resized_width"]),
                            int(letterbox["resized_height"]),
                        ),
                        interpolation=cv2.INTER_LINEAR,
                    )

                    metric_attribution = np.zeros(
                        metric_shape,
                        dtype=np.float32,
                    )

                    pad_x = int(letterbox["pad_x"])
                    pad_y = int(letterbox["pad_y"])
                    resized_height, resized_width = (
                        resized_attribution.shape
                    )

                    metric_attribution[
                        pad_y:pad_y + resized_height,
                        pad_x:pad_x + resized_width,
                    ] = resized_attribution

                elif metric_attribution.shape != metric_shape:
                    raise ValueError(
                        "Attribution shape does not match either "
                        "the original image or model input: "
                        f"{metric_attribution.shape}."
                    )

                original_target = context["xai_target"]
                original_bbox = original_target.bbox

                if original_bbox is None:
                    raise ValueError(
                        "Detection metrics require a target bbox."
                    )

                letterbox = context["letterbox"]
                scale = float(letterbox["scale"])
                pad_x = float(letterbox["pad_x"])
                pad_y = float(letterbox["pad_y"])

                x1, y1, x2, y2 = original_bbox

                metric_target = XAITarget(
                    task="detection",
                    class_id=original_target.class_id,
                    score=original_target.score,
                    detection_index=original_target.detection_index,
                    bbox=[
                        x1 * scale + pad_x,
                        y1 * scale + pad_y,
                        x2 * scale + pad_x,
                        y2 * scale + pad_y,
                    ],
                )

                metric_iou_threshold = float(
                    metrics_config.get(
                        "iou_threshold",
                        0.5,
                    )
                )

                deletion_result = compute_detection_deletion(
                    predictor=metric_predictor,
                    input_tensor=metric_input_tensor,
                    attribution=attribution,
                    target=metric_target,
                    steps=steps,
                    baseline_value=baseline_value,
                    iou_threshold=metric_iou_threshold,
                )

                insertion_result = compute_detection_insertion(
                    predictor=metric_predictor,
                    input_tensor=metric_input_tensor,
                    attribution=attribution,
                    target=metric_target,
                    steps=steps,
                    baseline_value=baseline_value,
                    iou_threshold=metric_iou_threshold,
                )

            else:

                deletion_result = compute_deletion(
                    model=model,
                    input_tensor=input_data,
                    attribution=attribution,
                    target=prediction.class_id,
                    steps=steps,
                    baseline_value=baseline_value,
                )

                insertion_result = compute_insertion(
                    model=model,
                    input_tensor=input_data,
                    attribution=attribution,
                    target=prediction.class_id,
                    steps=steps,
                    baseline_value=baseline_value,
                )

            metrics_path = (
                self.output_dir
                / f"{method}_faithfulness.png"
            )

            plot_faithfulness_curves(
                deletion_fractions=(
                    deletion_result.fractions
                ),
                deletion_confidences=(
                    deletion_result.confidences
                ),
                deletion_auc=(
                    deletion_result.auc
                ),
                insertion_fractions=(
                    insertion_result.fractions
                ),
                insertion_confidences=(
                    insertion_result.confidences
                ),
                insertion_auc=(
                    insertion_result.auc
                ),
                output_path=metrics_path,
            )

        result = {
            "method": explanation.method,
            "target": explanation.target,
            "confidence": explanation.confidence,
            "runtime_ms": explanation.runtime_ms,
            "attribution_shape": list(
                explanation.attribution.shape
            ),
            "artifacts": {
                "explanation": (
                    visualization_path.name
                ),
            },
            "metrics": None,
            "metadata": explanation.metadata,
        }

        if (
            deletion_result is not None
            and insertion_result is not None
        ):

            result["metrics"] = {
                "deletion_auc": (
                    deletion_result.auc
                ),
                "insertion_auc": (
                    insertion_result.auc
                ),
                "steps": steps,
                "baseline_value": (
                    baseline_value
                ),
            }

            if metric_iou_threshold is not None:
                result["metrics"]["iou_threshold"] = (
                    metric_iou_threshold
                )

            result["artifacts"][
                "faithfulness"
            ] = metrics_path.name

        return result

    # ============================================================
    # Detection heatmap restore
    # ============================================================

    @staticmethod
    def _restore_detection_heatmap(
        heatmap: np.ndarray,
        letterbox: dict[str, Any],
        output_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Map model-input CAM back to original-image resolution.

        No bbox masking is applied here.
        """

        if heatmap.ndim != 2:
            raise ValueError(
                "Detection heatmap must be 2D."
            )

        input_width = int(
            letterbox["input_width"]
        )

        input_height = int(
            letterbox["input_height"]
        )

        pad_x = int(
            letterbox["pad_x"]
        )

        pad_y = int(
            letterbox["pad_y"]
        )

        resized_width = int(
            letterbox["resized_width"]
        )

        resized_height = int(
            letterbox["resized_height"]
        )

        cam = cv2.resize(
            heatmap.astype(
                np.float32
            ),
            (
                input_width,
                input_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        y1 = pad_y
        y2 = (
            pad_y
            + resized_height
        )

        x1 = pad_x
        x2 = (
            pad_x
            + resized_width
        )

        cam = cam[
            y1:y2,
            x1:x2,
        ]

        output_width, output_height = (
            output_size
        )

        cam = cv2.resize(
            cam,
            (
                output_width,
                output_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        cam_min = float(
            cam.min()
        )

        cam_max = float(
            cam.max()
        )

        if (
            cam_max
            - cam_min
            > 1e-8
        ):
            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        return np.clip(
            cam,
            0.0,
            1.0,
        )

    # ============================================================
    # CSV
    # ============================================================

    def _save_comparison_csv(
        self,
        explanations: dict[str, Any],
    ) -> Path:

        output_path = (
            self.output_dir
            / "comparison.csv"
        )

        rows = []

        for method, result in (
            explanations.items()
        ):

            metrics = (
                result.get(
                    "metrics"
                )
                or {}
            )

            rows.append(
                {
                    "method": method,
                    "runtime_ms": (
                        result[
                            "runtime_ms"
                        ]
                    ),
                    "confidence": (
                        result[
                            "confidence"
                        ]
                    ),
                    "deletion_auc": (
                        metrics.get(
                            "deletion_auc"
                        )
                    ),
                    "insertion_auc": (
                        metrics.get(
                            "insertion_auc"
                        )
                    ),
                }
            )

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "method",
                    "runtime_ms",
                    "confidence",
                    "deletion_auc",
                    "insertion_auc",
                ],
            )

            writer.writeheader()
            writer.writerows(
                rows
            )

        return output_path

    # ============================================================
    # Runtime plot
    # ============================================================

    def _save_runtime_plot(
        self,
        explanations: dict[str, Any],
    ) -> Path:

        methods = list(
            explanations.keys()
        )

        runtimes = [
            explanations[
                method
            ][
                "runtime_ms"
            ]
            for method in methods
        ]

        output_path = (
            self.output_dir
            / "runtime_comparison.png"
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            methods,
            runtimes,
        )

        plt.ylabel(
            "Runtime (ms)"
        )

        plt.xlabel(
            "XAI method"
        )

        plt.title(
            "XAI Runtime Comparison"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

        return output_path

    # ============================================================
    # Faithfulness plot
    # ============================================================

    def _save_faithfulness_plot(
        self,
        explanations: dict[str, Any],
    ) -> Path:

        valid_methods = [
            method
            for method in explanations
            if explanations[
                method
            ][
                "metrics"
            ] is not None
        ]

        output_path = (
            self.output_dir
            / "faithfulness_comparison.png"
        )

        plt.figure(
            figsize=(9, 5)
        )

        if valid_methods:

            deletion = [
                explanations[
                    method
                ][
                    "metrics"
                ][
                    "deletion_auc"
                ]
                for method in valid_methods
            ]

            insertion = [
                explanations[
                    method
                ][
                    "metrics"
                ][
                    "insertion_auc"
                ]
                for method in valid_methods
            ]

            x = np.arange(
                len(valid_methods)
            )

            width = 0.35

            plt.bar(
                x - width / 2,
                deletion,
                width,
                label="Deletion AUC",
            )

            plt.bar(
                x + width / 2,
                insertion,
                width,
                label="Insertion AUC",
            )

            plt.xticks(
                x,
                valid_methods,
            )

            plt.legend()

        else:

            plt.text(
                0.5,
                0.5,
                "Faithfulness metrics "
                "are disabled for this task.",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )

            plt.xticks([])

        plt.ylabel(
            "AUC"
        )

        plt.xlabel(
            "XAI method"
        )

        plt.title(
            "XAI Faithfulness Comparison"
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

        return output_path

    # ============================================================
    # Main
    # ============================================================

    def run(
        self,
    ) -> dict[str, Any]:

        output_dir = (
            self._create_output_dir()
        )

        self._save_config(
            output_dir
        )

        print("=" * 60)
        print(
            "UAV-XAI EXPERIMENT"
        )
        print("=" * 60)

        print(
            f"Config: "
            f"{self.config_path}"
        )

        print(
            f"Device: "
            f"{self.device}"
        )

        print(
            f"Output: "
            f"{output_dir}"
        )

        # --------------------------------------------------------
        # 1. Model
        # --------------------------------------------------------

        print()
        print(
            "[1/6] Loading model..."
        )

        model_info = (
            self._load_model()
        )

        model = model_info[
            "model"
        ]

        weights = model_info[
            "weights"
        ]

        task = model_info[
            "task"
        ]

        model_name = model_info[
            "name"
        ]

        print(
            f"      Model: "
            f"{model_name}"
        )

        # --------------------------------------------------------
        # 2. Image
        # --------------------------------------------------------

        print()
        print(
            "[2/6] Loading image..."
        )

        (
            original_image,
            input_data,
        ) = self._load_image(
            weights=weights,
            task=task,
        )

        print(
            f"      Image size: "
            f"{original_image.size}"
        )

        # --------------------------------------------------------
        # 3. Prediction
        # --------------------------------------------------------

        print()
        print(
            "[3/6] Running prediction..."
        )

        predictor = ModelPredictor(
            model=model,
            task=task,
        )

        prediction = (
            predictor.predict(
                input_data
            )
        )

        predicted_class = None
        categories = None

        if task == "classification":

            categories = (
                weights.meta[
                    "categories"
                ]
            )

            predicted_class = (
                categories[
                    prediction.class_id
                ]
            )

            print(
                f"      Class: "
                f"{predicted_class}"
            )

            print(
                f"      Class ID: "
                f"{prediction.class_id}"
            )

            print(
                f"      Confidence: "
                f"{prediction.confidence:.4f}"
            )

        elif task == "detection":

            detections = (
                prediction.detections
                or []
            )

            if not detections:
                raise ValueError(
                    "YOLO produced "
                    "no detections."
                )

            for (
                index,
                detection,
            ) in enumerate(
                detections,
                start=1,
            ):

                print(
                    f"      Detection {index}: "
                    f"{detection.class_name} "
                    f"confidence="
                    f"{detection.confidence:.4f} "
                    f"bbox="
                    f"{detection.bbox}"
                )

        else:

            raise ValueError(
                f"Unsupported task: "
                f"{task}"
            )

        # --------------------------------------------------------
        # XAI target
        # --------------------------------------------------------

        target_config = (
            self.config.get(
                "target",
                {},
            )
        )

        target_type = (
            target_config.get(
                "type"
            )
        )

        detection_index = int(
            target_config.get(
                "index",
                0,
            )
        )

        xai_target = (
            TargetResolver.resolve(
                prediction=prediction,
                target_type=target_type,
                detection_index=(
                    detection_index
                ),
            )
        )

        # --------------------------------------------------------
        # XAI input
        # --------------------------------------------------------

        letterbox = None

        if task == "detection":

            (
                xai_input,
                letterbox,
            ) = (
                _prepare_yolo_xai_input(
                    original_image,
                    size=int(
                        self.config[
                            "xai"
                        ].get(
                            "image_size",
                            640,
                        )
                    ),
                )
            )

            xai_input = (
                xai_input
                .detach()
                .clone()
                .requires_grad_(True)
                .to(self.device)
            )

        else:

            xai_input = (
                input_data
            )

        print(
            f"      XAI input: "
            f"shape="
            f"{tuple(xai_input.shape)} "
            f"requires_grad="
            f"{xai_input.requires_grad}"
        )

        # --------------------------------------------------------
        # 4. XAI
        # --------------------------------------------------------

        print()
        print(
            "[4/6] Generating explanations..."
        )

        engine = (
            create_default_engine()
        )

        methods = (
            self._get_methods()
        )

        context = (
            self._build_xai_context(
                model=model,
                weights=weights,
                original_image=(
                    original_image
                ),
                input_data=input_data,
                xai_input=xai_input,
                xai_target=xai_target,
                task=task,
                letterbox=letterbox,
            )
        )

        explanations: dict[
            str,
            Any,
        ] = {}

        try:

            for (
                index,
                method,
            ) in enumerate(
                methods,
                start=1,
            ):

                print()

                print(
                    f"      [{index}/{len(methods)}] "
                    f"{method}"
                )

                method_result = (
                    self._run_explanation(
                        engine=engine,
                        model=model,
                        prediction=(
                            prediction
                        ),
                        original_image=(
                            original_image
                        ),
                        input_data=input_data,
                        context=context,
                        method=method,
                    )
                )

                explanations[
                    method
                ] = method_result

                print(
                    f"          Runtime: "
                    f"{method_result['runtime_ms']:.2f} ms"
                )

                if (
                    method_result[
                        "metrics"
                    ]
                    is not None
                ):

                    print(
                        f"          Deletion AUC: "
                        f"{method_result['metrics']['deletion_auc']:.4f}"
                    )

                    print(
                        f"          Insertion AUC: "
                        f"{method_result['metrics']['insertion_auc']:.4f}"
                    )

            # ----------------------------------------------------
            # 5. Summary
            # ----------------------------------------------------

            print()
            print(
                "[5/6] Saving summary..."
            )

            comparison_csv = (
                self._save_comparison_csv(
                    explanations
                )
            )

            runtime_plot = (
                self._save_runtime_plot(
                    explanations
                )
            )

            faithfulness_plot = (
                self._save_faithfulness_plot(
                    explanations
                )
            )

            prediction_json: dict[
                str,
                Any,
            ] = {
                "class_id": (
                    prediction.class_id
                ),
                "confidence": (
                    prediction.confidence
                ),
                "task": (
                    prediction.task
                ),
            }

            if task == "classification":

                prediction_json[
                    "class_name"
                ] = predicted_class

                prediction_json[
                    "top5"
                ] = [
                    {
                        "class_id": int(
                            index
                        ),
                        "class_name": (
                            categories[
                                int(index)
                            ]
                        ),
                        "confidence": float(
                            prediction.probabilities[
                                index
                            ]
                        ),
                    }
                    for index in (
                        prediction.probabilities
                        .argsort()[::-1][:5]
                    )
                ]

            else:

                prediction_json[
                    "detections"
                ] = [
                    {
                        "class_id": (
                            detection.class_id
                        ),
                        "class_name": (
                            detection.class_name
                        ),
                        "confidence": (
                            detection.confidence
                        ),
                        "bbox": (
                            detection.bbox
                        ),
                    }
                    for detection in (
                        prediction.detections
                        or []
                    )
                ]

            result: dict[
                str,
                Any,
            ] = {
                "experiment": (
                    self.config[
                        "experiment"
                    ][
                        "name"
                    ]
                ),
                "timestamp": (
                    datetime.now().isoformat()
                ),
                "device": str(
                    self.device
                ),
                "model": {
                    "name": (
                        model_name
                    ),
                    "task": task,
                    "weights": (
                        "DEFAULT"
                        if weights is not None
                        else self.config[
                            "model"
                        ].get(
                            "weights"
                        )
                    ),
                },
                "input": {
                    "image_path": (
                        self.config[
                            "image"
                        ][
                            "path"
                        ]
                    ),
                    "original_size": list(
                        original_image.size
                    ),
                },
                "prediction": (
                    prediction_json
                ),
                "target": {
                    "task": (
                        xai_target.task
                    ),
                    "class_id": (
                        xai_target.class_id
                    ),
                    "score": (
                        xai_target.score
                    ),
                    "detection_index": (
                        xai_target.detection_index
                    ),
                    "bbox": (
                        xai_target.bbox
                    ),
                },
                "explanations": (
                    explanations
                ),
                "artifacts": {
                    "comparison_csv": (
                        comparison_csv.name
                    ),
                    "runtime_comparison": (
                        runtime_plot.name
                    ),
                    "faithfulness_comparison": (
                        faithfulness_plot.name
                    ),
                },
            }

            if letterbox is not None:

                result[
                    "xai_preprocessing"
                ] = {
                    "letterbox": (
                        letterbox
                    )
                }

            result_path = (
                output_dir
                / "result.json"
            )

            with result_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    result,
                    file,
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                )

            print(
                f"      Result: "
                f"{result_path}"
            )

            print(
                f"      Comparison: "
                f"{comparison_csv}"
            )

            print(
                f"      Runtime plot: "
                f"{runtime_plot}"
            )

            print(
                f"      Faithfulness plot: "
                f"{faithfulness_plot}"
            )

            # ----------------------------------------------------
            # 6. Final
            # ----------------------------------------------------

            print()
            print(
                "[6/6] Experiment complete"
            )

            print("=" * 60)

            print(
                "EXPERIMENT COMPLETE"
            )

            print("=" * 60)

            print(
                f"Result: "
                f"{result_path}"
            )

            return result

        finally:

            engine.close()