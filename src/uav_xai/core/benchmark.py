import csv
import json
import matplotlib.pyplot as plt
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
from .runner import (
    _json_default,
    _resolve_module,
)
from ..core import create_default_engine
from ..metrics import (
    compute_deletion,
    compute_insertion,
)
from ..utils import (
    overlay_heatmap,
    plot_faithfulness_curves,
    save_image,
)


class BenchmarkRunner:
    """
    Run repeated XAI benchmarks over multiple images.

    The model is loaded once and reused for all images/runs.
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
        ]["name"]

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        root = Path(
            self.config["output"]["root"]
        )

        output_dir = (
            root
            / f"{experiment_name}_{timestamp}"
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
        image_path: Path,
        weights: ResNet18_Weights,
    ):
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

    def _get_paths(self) -> list[Path]:
        input_config = self.config[
            "input"
        ]

        paths = input_config.get(
            "paths"
        )

        if not isinstance(
            paths,
            list,
        ):
            raise ValueError(
                "input.paths must be a list."
            )

        if not paths:
            raise ValueError(
                "input.paths cannot be empty."
            )

        return [
            Path(path)
            for path in paths
        ]

    def _get_methods(self) -> list[str]:
        methods = self.config[
            "xai"
        ].get(
            "methods"
        )

        if not isinstance(
            methods,
            list,
        ):
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

    def _build_context(
        self,
        model: torch.nn.Module,
        weights: ResNet18_Weights,
        image: Image.Image,
        input_tensor: torch.Tensor,
    ) -> dict[str, Any]:

        xai_config = self.config[
            "xai"
        ]

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
            "input_type": self.config[
                "input"
            ].get(
                "type",
                "image",
            ),
            "xai_config": xai_config,
            "target_layer": target_layer,
            "preprocess": preprocess,
            "input_tensor": input_tensor,
            "original_image": image,
        }

    def _run_method(
        self,
        engine,
        model: torch.nn.Module,
        weights: ResNet18_Weights,
        image: Image.Image,
        input_tensor: torch.Tensor,
        prediction: Any,
        context: dict[str, Any],
        method: str,
        output_dir: Path,
    ) -> dict[str, Any]:

        start = torch.cuda.Event(
            enable_timing=True
        ) if self.device.type == "cuda" else None

        if start is not None:
            end = torch.cuda.Event(
                enable_timing=True
            )
            start.record()

        else:
            end = None

        wall_start = (
            __import__("time")
            .perf_counter()
        )

        explanation = engine.explain(
            input_type=context[
                "input_type"
            ],
            method=method,
            model=model,
            input_data=input_tensor,
            target=prediction.class_id,
            context=context,
        )

        wall_runtime_ms = (
            __import__("time").perf_counter()
            - wall_start
        ) * 1000.0

        if start is not None and end is not None:
            end.record()
            torch.cuda.synchronize()
            wall_runtime_ms = (
                start.elapsed_time(end)
            )

        visualization_path = (
            output_dir
            / f"{method}.png"
        )

        visualization = overlay_heatmap(
            image=image,
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

        metrics = None

        if metrics_config.get(
            "enabled",
            True,
        ):
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

            deletion = compute_deletion(
                model=model,
                input_tensor=input_tensor,
                attribution=explanation.attribution,
                target=prediction.class_id,
                steps=steps,
                baseline_value=baseline_value,
            )

            insertion = compute_insertion(
                model=model,
                input_tensor=input_tensor,
                attribution=explanation.attribution,
                target=prediction.class_id,
                steps=steps,
                baseline_value=baseline_value,
            )

            faithfulness_path = (
                output_dir
                / f"{method}_faithfulness.png"
            )

            plot_faithfulness_curves(
                deletion_fractions=(
                    deletion.fractions
                ),
                deletion_confidences=(
                    deletion.confidences
                ),
                deletion_auc=deletion.auc,
                insertion_fractions=(
                    insertion.fractions
                ),
                insertion_confidences=(
                    insertion.confidences
                ),
                insertion_auc=insertion.auc,
                output_path=faithfulness_path,
            )

            metrics = {
                "deletion_auc": deletion.auc,
                "insertion_auc": insertion.auc,
            }

        return {
            "method": method,
            "runtime_ms": explanation.runtime_ms,
            "wall_runtime_ms": wall_runtime_ms,
            "confidence": explanation.confidence,
            "metrics": metrics,
            "artifacts": {
                "explanation": visualization_path.name,
            },
        }

    def _save_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> None:

        csv_path = (
            self.output_dir
            / "comparison.csv"
        )

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "image",
                    "run",
                    "method",
                    "runtime_ms",
                    "wall_runtime_ms",
                    "confidence",
                    "deletion_auc",
                    "insertion_auc",
                ],
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(row)

    def _save_summary(
        self,
        rows: list[dict[str, Any]],
    ) -> None:

        groups: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for row in rows:
            groups.setdefault(
                row["method"],
                [],
            ).append(
                row
            )

        summary = []

        for method, values in groups.items():

            def stats(
                key: str,
            ) -> tuple[float | None, float | None]:

                data = [
                    float(value[key])
                    for value in values
                    if value[key] is not None
                ]

                if not data:
                    return None, None

                return (
                    float(np.mean(data)),
                    float(np.std(data)),
                )

            runtime_mean, runtime_std = stats(
                "runtime_ms"
            )

            wall_mean, wall_std = stats(
                "wall_runtime_ms"
            )

            deletion_mean, deletion_std = stats(
                "deletion_auc"
            )

            insertion_mean, insertion_std = stats(
                "insertion_auc"
            )

            summary.append(
                {
                    "method": method,
                    "runs": len(values),
                    "runtime_mean_ms": runtime_mean,
                    "runtime_std_ms": runtime_std,
                    "wall_runtime_mean_ms": wall_mean,
                    "wall_runtime_std_ms": wall_std,
                    "deletion_auc_mean": deletion_mean,
                    "deletion_auc_std": deletion_std,
                    "insertion_auc_mean": insertion_mean,
                    "insertion_auc_std": insertion_std,
                }
            )

        summary_path = (
            self.output_dir
            / "summary.json"
        )

        with summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )

    def _save_runtime_plot(
        self,
        rows: list[dict[str, Any]],
    ) -> Path:
        """
        Save mean ± std runtime comparison.
        """

        groups: dict[str, list[float]] = {}

        for row in rows:
            groups.setdefault(
                row["method"],
                [],
            ).append(
                float(
                    row["runtime_ms"]
                )
            )

        methods = list(
            groups.keys()
        )

        means = [
            float(
                np.mean(
                    groups[method]
                )
            )
            for method in methods
        ]

        stds = [
            float(
                np.std(
                    groups[method]
                )
            )
            for method in methods
        ]

        output_path = (
            self.output_dir
            / "runtime_comparison.png"
        )

        plt.figure(
            figsize=(9, 5)
        )

        plt.bar(
            methods,
            means,
            yerr=stds,
            capsize=5,
        )

        plt.ylabel(
            "Runtime (ms)"
        )

        plt.xlabel(
            "XAI method"
        )

        plt.title(
            "XAI Runtime: Mean ± Std"
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
        rows: list[dict[str, Any]],
    ) -> Path:
        """
        Save mean ± std deletion/insertion AUC comparison.
        """

        deletion_groups: dict[
            str,
            list[float],
        ] = {}

        insertion_groups: dict[
            str,
            list[float],
        ] = {}

        for row in rows:

            method = row["method"]

            if row["deletion_auc"] is not None:
                deletion_groups.setdefault(
                    method,
                    [],
                ).append(
                    float(
                        row["deletion_auc"]
                    )
                )

            if row["insertion_auc"] is not None:
                insertion_groups.setdefault(
                    method,
                    [],
                ).append(
                    float(
                        row["insertion_auc"]
                    )
                )

        methods = list(
            dict.fromkeys(
                list(deletion_groups.keys())
                + list(insertion_groups.keys())
            )
        )

        deletion_means = [
            float(
                np.mean(
                    deletion_groups.get(
                        method,
                        [0.0],
                    )
                )
            )
            for method in methods
        ]

        deletion_stds = [
            float(
                np.std(
                    deletion_groups.get(
                        method,
                        [0.0],
                    )
                )
            )
            for method in methods
        ]

        insertion_means = [
            float(
                np.mean(
                    insertion_groups.get(
                        method,
                        [0.0],
                    )
                )
            )
            for method in methods
        ]

        insertion_stds = [
            float(
                np.std(
                    insertion_groups.get(
                        method,
                        [0.0],
                    )
                )
            )
            for method in methods
        ]

        x = np.arange(
            len(methods)
        )

        width = 0.35

        output_path = (
            self.output_dir
            / "faithfulness_comparison.png"
        )

        plt.figure(
            figsize=(9, 5)
        )

        plt.bar(
            x - width / 2,
            deletion_means,
            width,
            yerr=deletion_stds,
            capsize=5,
            label="Deletion AUC",
        )

        plt.bar(
            x + width / 2,
            insertion_means,
            width,
            yerr=insertion_stds,
            capsize=5,
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
            "XAI Faithfulness: Mean ± Std"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

        return output_path

    def _print_summary(
        self,
        rows: list[dict[str, Any]],
    ) -> None:
        """
        Print aggregated benchmark statistics.
        """

        groups: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for row in rows:
            groups.setdefault(
                row["method"],
                [],
            ).append(
                row
            )

        print()
        print(
            "=" * 90
        )
        print(
            "BENCHMARK SUMMARY"
        )
        print(
            "=" * 90
        )

        print(
            f"{'METHOD':<12}"
            f"{'RUNTIME ms':>20}"
            f"{'DELETION':>20}"
            f"{'INSERTION':>20}"
        )

        print(
            "-" * 90
        )

        for method, values in groups.items():

            runtime = np.asarray(
                [
                    float(
                        value["runtime_ms"]
                    )
                    for value in values
                ],
                dtype=np.float64,
            )

            deletion = np.asarray(
                [
                    float(
                        value["deletion_auc"]
                    )
                    for value in values
                    if value["deletion_auc"]
                    is not None
                ],
                dtype=np.float64,
            )

            insertion = np.asarray(
                [
                    float(
                        value["insertion_auc"]
                    )
                    for value in values
                    if value["insertion_auc"]
                    is not None
                ],
                dtype=np.float64,
            )

            runtime_text = (
                f"{runtime.mean():.2f} ± "
                f"{runtime.std():.2f}"
            )

            deletion_text = (
                f"{deletion.mean():.4f} ± "
                f"{deletion.std():.4f}"
                if deletion.size
                else "n/a"
            )

            insertion_text = (
                f"{insertion.mean():.4f} ± "
                f"{insertion.std():.4f}"
                if insertion.size
                else "n/a"
            )

            print(
                f"{method:<12}"
                f"{runtime_text:>20}"
                f"{deletion_text:>20}"
                f"{insertion_text:>20}"
            )

        print(
            "=" * 90
        )

    def run(self) -> dict[str, Any]:
        output_dir = (
            self._create_output_dir()
        )

        shutil.copy2(
            self.config_path,
            output_dir / "config.yaml",
        )

        model, weights = self._load_model()

        paths = self._get_paths()
        methods = self._get_methods()

        runs = int(
            self.config[
                "benchmark"
            ].get(
                "runs",
                1,
            )
        )

        if runs < 1:
            raise ValueError(
                "benchmark.runs must be >= 1."
            )

        engine = create_default_engine()

        rows: list[dict[str, Any]] = []

        for image_path in paths:

            for run_number in range(
                1,
                runs + 1,
            ):

                image, input_tensor = (
                    self._load_image(
                        image_path,
                        weights,
                    )
                )

                predictor = ModelPredictor(
                    model
                )

                prediction = (
                    predictor.predict(
                        input_tensor
                    )
                )

                context = self._build_context(
                    model=model,
                    weights=weights,
                    image=image,
                    input_tensor=input_tensor,
                )

                run_dir = (
                    output_dir
                    / image_path.stem
                    / f"run_{run_number:02d}"
                )

                run_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                for method in methods:

                    result = self._run_method(
                        engine=engine,
                        model=model,
                        weights=weights,
                        image=image,
                        input_tensor=input_tensor,
                        prediction=prediction,
                        context=context,
                        method=method,
                        output_dir=run_dir,
                    )

                    metrics = (
                        result["metrics"]
                        or {}
                    )

                    rows.append(
                        {
                            "image": image_path.name,
                            "run": run_number,
                            "method": method,
                            "runtime_ms": result[
                                "runtime_ms"
                            ],
                            "wall_runtime_ms": result[
                                "wall_runtime_ms"
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

        self._save_rows(
            rows
        )

        self._save_summary(
            rows
        )

        runtime_plot = (
            self._save_runtime_plot(
                rows
            )
        )

        faithfulness_plot = (
            self._save_faithfulness_plot(
                rows
            )
        )

        self._print_summary(
            rows
        )

        engine.close()

        return {
            "experiment": self.config[
                "experiment"
            ]["name"],
            "images": [
                str(path)
                for path in paths
            ],
            "runs": runs,
            "methods": methods,
            "artifacts": {
                "comparison": "comparison.csv",
                "summary": "summary.json",
                "runtime_plot": runtime_plot.name,
                "faithfulness_plot": (
                    faithfulness_plot.name
                ),
            },
        }