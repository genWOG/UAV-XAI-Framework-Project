import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

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
from ..metrics import (
    compute_deletion,
    compute_insertion,
)
from ..utils import (
    overlay_heatmap,
    plot_faithfulness_curves,
    save_image,
)
from ..core import create_default_engine


def _resolve_module(
    root: torch.nn.Module,
    path: str,
) -> torch.nn.Module:
    """
    Resolve a PyTorch module from a dotted path.

    Example
    -------
    layer4[-1].conv2

    Parameters
    ----------
    root:
        Root PyTorch module.

    path:
        Module path.

    Returns
    -------
    torch.nn.Module
    """

    current: Any = root

    for part in path.split("."):

        if "[" in part and part.endswith("]"):
            name, index_text = part.split("[", 1)

            index = int(
                index_text[:-1]
            )

            current = getattr(
                current,
                name,
            )

            current = current[index]

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
            f"is not a torch.nn.Module."
        )

    return current


def _json_default(value: Any) -> Any:
    """
    Convert numpy/PyTorch objects into JSON-compatible values.
    """

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
        return value.detach().cpu().tolist()

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


class ExperimentRunner:
    """
    Run a complete XAI experiment from a YAML configuration.
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

        self.output_dir: Path | None = None

    def _create_output_dir(self) -> Path:
        experiment_name = self.config[
            "experiment"
        ][
            "name"
        ]

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        root = Path(
            self.config["output"]["root"]
        )

        output_dir = root / (
            f"{experiment_name}_{timestamp}"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        self.output_dir = output_dir

        return output_dir

    def _load_model(self):
        model_config = self.config[
            "model"
        ]

        model_name = model_config[
            "name"
        ].lower()

        if model_name != "resnet18":
            raise ValueError(
                f"Unsupported model: {model_name}. "
                "Currently supported: resnet18."
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

        return model, weights

    def _load_image(
        self,
        weights: ResNet18_Weights,
    ):
        image_path = Path(
            self.config["image"]["path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        transform = weights.transforms()

        input_tensor = transform(
            image
        ).unsqueeze(0)

        input_tensor = input_tensor.to(
            self.device
        )

        return image, input_tensor

    def _get_methods(self) -> list[str]:
        """
        Return configured XAI methods.

        Supports both:
            xai.method: grad_cam

        and:
            xai.methods:
            - grad_cam
            - lime
            - shap
            - lrp
        """

        xai_config = self.config["xai"]

        if "methods" in xai_config:
            methods = xai_config["methods"]

            if not isinstance(methods, list):
                raise ValueError(
                    "xai.methods must be a list."
                )

            if not methods:
                raise ValueError(
                    "xai.methods cannot be empty."
                )

            return [
                str(method).strip().lower()
                for method in methods
            ]

        if "method" in xai_config:
            return [
                str(
                    xai_config["method"]
                ).strip().lower()
            ]

        raise ValueError(
            "XAI configuration must contain "
            "'method' or 'methods'."
        )


    def _build_xai_context(
        self,
        model: torch.nn.Module,
        weights: ResNet18_Weights,
        original_image: Image.Image,
        input_tensor: torch.Tensor,
    ) -> dict[str, Any]:
        """
        Build shared context used by explainers.
        """

        xai_config = self.config["xai"]

        target_layer = None

        target_layer_path = xai_config.get(
            "target_layer"
        )

        if target_layer_path:
            target_layer = _resolve_module(
                model,
                target_layer_path,
            )

        transform = weights.transforms()

        def preprocess(
            rgb_image: np.ndarray,
        ) -> torch.Tensor:
            pil_image = Image.fromarray(
                rgb_image.astype(
                    np.uint8
                )
            )

            return transform(
                pil_image
            )

        return {
            "model": model,
            "weights": weights,
            "input_type": xai_config.get(
                "input_type",
                "image",
            ),
            "xai_config": xai_config,
            "target_layer": target_layer,
            "preprocess": preprocess,
            "input_tensor": input_tensor,
            "original_image": original_image,
        }

    def _save_config(
        self,
        output_dir: Path,
    ) -> None:
        destination = (
            output_dir / "config.yaml"
        )

        shutil.copy2(
            self.config_path,
            destination,
        )

    def _run_explanation(
        self,
        engine,
        model: torch.nn.Module,
        prediction: Any,
        original_image: Image.Image,
        input_tensor: torch.Tensor,
        context: dict[str, Any],
        method: str,
    ) -> dict[str, Any]:
        """
        Run one XAI method and save its artifacts.
        """

        input_type = context["input_type"]

        explanation = engine.explain(
            input_type=input_type,
            method=method,
            model=model,
            input_data=input_tensor,
            target=prediction.class_id,
            context=context,
        )

        visualization_path = (
            self.output_dir
            / f"{method}.png"
        )

        visualization = overlay_heatmap(
            image=original_image,
            heatmap=explanation.attribution,
            alpha=0.45,
        )

        save_image(
            visualization,
            visualization_path,
        )

        metrics_config = self.config.get(
            "metrics",
            {},
        )

        metrics_enabled = metrics_config.get(
            "enabled",
            True,
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

            deletion_result = compute_deletion(
                model=model,
                input_tensor=input_tensor,
                attribution=explanation.attribution,
                target=prediction.class_id,
                steps=steps,
                baseline_value=baseline_value,
            )

            insertion_result = compute_insertion(
                model=model,
                input_tensor=input_tensor,
                attribution=explanation.attribution,
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

            result["artifacts"][
                "faithfulness"
            ] = metrics_path.name

        return result

    def _save_comparison_csv(
        self,
        explanations: dict[str, Any],
    ) -> Path:
        """
        Save aggregated XAI benchmark results to CSV.
        """

        output_path = (
            self.output_dir
            / "comparison.csv"
        )

        rows = []

        for method, result in explanations.items():
            metrics = result.get(
                "metrics"
            ) or {}

            rows.append(
                {
                    "method": method,
                    "runtime_ms": result[
                        "runtime_ms"
                    ],
                    "confidence": result[
                        "confidence"
                    ],
                    "deletion_auc": metrics.get(
                        "deletion_auc"
                    ),
                    "insertion_auc": metrics.get(
                        "insertion_auc"
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
            writer.writerows(rows)

        return output_path

    def _save_runtime_plot(
        self,
        explanations: dict[str, Any],
    ) -> Path:
        """
        Save XAI runtime comparison plot.
        """

        methods = list(
            explanations.keys()
        )

        runtimes = [
            explanations[method][
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

    def _save_faithfulness_plot(
        self,
        explanations: dict[str, Any],
    ) -> Path:
        """
        Save deletion/insertion AUC comparison.
        """

        methods = list(
            explanations.keys()
        )

        deletion = [
            explanations[method][
                "metrics"
            ]["deletion_auc"]
            for method in methods
            if explanations[method][
                "metrics"
            ] is not None
        ]

        insertion = [
            explanations[method][
                "metrics"
            ]["insertion_auc"]
            for method in methods
            if explanations[method][
                "metrics"
            ] is not None
        ]

        output_path = (
            self.output_dir
            / "faithfulness_comparison.png"
        )

        x = np.arange(
            len(methods)
        )

        width = 0.35

        plt.figure(
            figsize=(9, 5)
        )

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
            methods,
        )

        plt.ylabel(
            "AUC"
        )

        plt.xlabel(
            "XAI method"
        )

        plt.title(
            "XAI Faithfulness Comparison"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

        return output_path

    def run(self) -> dict[str, Any]:
        """
        Run a configured XAI experiment.

        Supports a single method or a list of methods.
        """

        output_dir = self._create_output_dir()

        self._save_config(
            output_dir
        )

        print("=" * 60)
        print("UAV-XAI EXPERIMENT")
        print("=" * 60)

        print(
            f"Config: {self.config_path}"
        )

        print(
            f"Device: {self.device}"
        )

        print(
            f"Output: {output_dir}"
        )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        print()
        print("[1/6] Loading model...")

        model, weights = self._load_model()

        print("      Model: ResNet-18")

        # --------------------------------------------------
        # Image
        # --------------------------------------------------

        print()
        print("[2/6] Loading image...")

        original_image, input_tensor = (
            self._load_image(weights)
        )

        print(
            f"      Image size: "
            f"{original_image.size}"
        )

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------

        print()
        print("[3/6] Running prediction...")

        predictor = ModelPredictor(
            model
        )

        prediction = predictor.predict(
            input_tensor
        )

        categories = weights.meta[
            "categories"
        ]

        predicted_class = categories[
            prediction.class_id
        ]

        print(
            f"      Class: {predicted_class}"
        )

        print(
            f"      Class ID: "
            f"{prediction.class_id}"
        )

        print(
            f"      Confidence: "
            f"{prediction.confidence:.4f}"
        )

        # --------------------------------------------------
        # XAI
        # --------------------------------------------------

        print()
        print("[4/6] Generating explanations...")

        engine = create_default_engine()

        methods = self._get_methods()

        context = self._build_xai_context(
            model=model,
            weights=weights,
            original_image=original_image,
            input_tensor=input_tensor,
        )

        explanations: dict[str, Any] = {}

        for index, method in enumerate(
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
                    prediction=prediction,
                    original_image=original_image,
                    input_tensor=input_tensor,
                    context=context,
                    method=method,
                )
            )

            explanations[method] = method_result

            print(
                f"          Runtime: "
                f"{method_result['runtime_ms']:.2f} ms"
            )

            if method_result["metrics"] is not None:
                print(
                    f"          Deletion AUC: "
                    f"{method_result['metrics']['deletion_auc']:.4f}"
                )

                print(
                    f"          Insertion AUC: "
                    f"{method_result['metrics']['insertion_auc']:.4f}"
                )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        print()
        print("[5/6] Saving summary...")

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

        result: dict[str, Any] = {
            "experiment": self.config[
                "experiment"
            ]["name"],
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "model": {
                "name": "resnet18",
                "weights": "DEFAULT",
            },
            "input": {
                "image_path": self.config[
                    "image"
                ]["path"],
                "original_size": list(
                    original_image.size
                ),
            },
            "prediction": {
                "class_id": prediction.class_id,
                "class_name": predicted_class,
                "confidence": prediction.confidence,
                "top5": [
                    {
                        "class_id": int(index),
                        "class_name": categories[
                            int(index)
                        ],
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
                ],
            },
            "explanations": explanations,
            "artifacts": {
                "comparison_csv": comparison_csv.name,
                "runtime_comparison": runtime_plot.name,
                "faithfulness_comparison": faithfulness_plot.name,
            },
        }

        result_path = (
            output_dir / "result.json"
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
            f"      Result: {result_path}"
        )
        print(
            f"      Comparison: {comparison_csv}"
        )

        print(
            f"      Runtime plot: {runtime_plot}"
        )

        print(
            f"      Faithfulness plot: "
            f"{faithfulness_plot}"
        )

        # --------------------------------------------------
        # Final
        # --------------------------------------------------

        print()
        print("[6/6] Experiment complete")

        print("=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)

        print(
            f"Result: {result_path}"
        )

        engine.close()

        return result