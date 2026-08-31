import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

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

    def run(self) -> dict[str, Any]:
        """
        Run the configured experiment.

        Returns
        -------
        dict
            Complete experiment result.
        """

        output_dir = (
            self._create_output_dir()
        )

        self._save_config(
            output_dir
        )

        print(
            "=" * 60
        )

        print(
            "UAV-XAI EXPERIMENT"
        )

        print(
            "=" * 60
        )

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
        print(
            "[1/6] Loading model..."
        )

        model, weights = (
            self._load_model()
        )

        print(
            "      Model: ResNet-18"
        )

        # --------------------------------------------------
        # Image
        # --------------------------------------------------

        print()
        print(
            "[2/6] Loading image..."
        )

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
        print(
            "[3/6] Running prediction..."
        )

        predictor = ModelPredictor(
            model
        )

        prediction = (
            predictor.predict(
                input_tensor
            )
        )

        categories = weights.meta[
            "categories"
        ]

        predicted_class = categories[
            prediction.class_id
        ]

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

        # --------------------------------------------------
        # XAI
        # --------------------------------------------------

        print()
        print(
            "[4/6] Generating explanation..."
        )

        engine = create_default_engine()

        method = self.config[
            "xai"
        ][
            "method"
        ]

        input_type = self.config["xai"].get(
            "input_type",
            "image",
        )

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
                rgb_image.astype(np.uint8)
            )

            return transform(
                pil_image
            )


        context = {
            "model": model,
            "weights": weights,
            "input_type": input_type,
            "xai_config": xai_config,
            "target_layer": target_layer,
            "preprocess": preprocess,
            "input_tensor": input_tensor,
            "original_image": original_image,
        }
        
        target_layer_path = self.config[
            "xai"
        ].get(
            "target_layer"
        )

        input_type = self.config["xai"].get(
            "input_type",
            "image",
        )

        target_layer = None

        explanation = engine.explain(
            input_type=input_type,
            method=method,
            model=model,
            input_data=input_tensor,
            target=prediction.class_id,
            context=context,
        )

        print(
            f"      Method: {method}"
        )

        print(
            f"      Runtime: "
            f"{explanation.runtime_ms:.2f} ms"
        )

        # --------------------------------------------------
        # Visualization
        # --------------------------------------------------

        print()
        print(
            "[5/6] Saving visualization..."
        )

        visualization_path = (
            output_dir
            / f"{method}.png"
        )

        visualization = (
            overlay_heatmap(
                image=original_image,
                heatmap=explanation.attribution,
                alpha=0.45,
            )
        )

        save_image(
            visualization,
            visualization_path,
        )

        print(
            f"      Saved: {visualization_path}"
        )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        print()
        print(
            "[6/6] Computing metrics..."
        )

        metrics_config = self.config[
            "metrics"
        ]

        metrics_enabled = metrics_config.get(
            "enabled",
            True,
        )

        deletion_result = None
        insertion_result = None

        metrics_path = (
            output_dir
            / "faithfulness.png"
        )

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

            deletion_result = (
                compute_deletion(
                    model=model,
                    input_tensor=input_tensor,
                    attribution=explanation.attribution,
                    target=prediction.class_id,
                    steps=steps,
                    baseline_value=baseline_value,
                )
            )

            insertion_result = (
                compute_insertion(
                    model=model,
                    input_tensor=input_tensor,
                    attribution=explanation.attribution,
                    target=prediction.class_id,
                    steps=steps,
                    baseline_value=baseline_value,
                )
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

            print(
                f"      Deletion AUC: "
                f"{deletion_result.auc:.4f}"
            )

            print(
                f"      Insertion AUC: "
                f"{insertion_result.auc:.4f}"
            )

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        result: dict[str, Any] = {
            "experiment": self.config[
                "experiment"
            ][
                "name"
            ],
            "timestamp": datetime.now().isoformat(),
            "device": str(
                self.device
            ),
            "model": {
                "name": "resnet18",
                "weights": "DEFAULT",
            },
            "input": {
                "image_path": self.config[
                    "image"
                ][
                    "path"
                ],
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
                        "class_id": int(
                            index
                        ),
                        "class_name": categories[
                            int(index)
                        ],
                        "confidence": float(
                            prediction.probabilities[
                                index
                            ]
                        ),
                    }
                    for index in prediction.probabilities.argsort()[
                        ::-1
                    ][
                        :5
                    ]
                ],
            },
            "explanation": {
                "method": explanation.method,
                "target": explanation.target,
                "confidence": explanation.confidence,
                "runtime_ms": explanation.runtime_ms,
                "attribution_shape": list(
                    explanation.attribution.shape
                ),
            },
            "metrics": None,
            "artifacts": {
                "explanation": str(
                    visualization_path.name
                ),
            },
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
            ] = str(
                metrics_path.name
            )

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

        print()
        print(
            "=" * 60
        )

        print(
            "EXPERIMENT COMPLETE"
        )

        print(
            "=" * 60
        )

        print(
            f"Result: {result_path}"
        )

        print(
            f"Explanation: {visualization_path}"
        )

        if metrics_enabled:
            print(
                f"Faithfulness: "
                f"{metrics_path}"
            )

        engine.close()

        return result