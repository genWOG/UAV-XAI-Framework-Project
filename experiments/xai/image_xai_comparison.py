import json
import time
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

from uav_xai import create_default_engine
from uav_xai.methods import LimeImageExplainer
from uav_xai.metrics import (
    compute_deletion,
    compute_insertion,
)
from uav_xai.utils import overlay_heatmap, save_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "images"
    / "test.jpg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "image_xai"
    / "comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def normalize_attribution(
    attribution: np.ndarray,
) -> np.ndarray:
    """
    Normalize attribution by absolute maximum.

    Returns values in [-1, 1].
    """

    attribution = np.asarray(
        attribution,
        dtype=np.float32,
    )

    max_abs = float(
        np.max(
            np.abs(attribution)
        )
    )

    if max_abs == 0.0:
        return np.zeros_like(
            attribution
        )

    return attribution / max_abs


def positive_attribution(
    attribution: np.ndarray,
) -> np.ndarray:
    """
    Keep only positive attribution for visualization.

    This is useful for comparing regions that support
    the selected target class.
    """

    normalized = normalize_attribution(
        attribution
    )

    return np.maximum(
        normalized,
        0.0,
    )


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    print("=" * 70)
    print("IMAGE XAI COMPARISON")
    print("Grad-CAM vs LIME")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights
    ).to(device)

    model.eval()

    print(
        "Model: ResNet-18"
    )

    # --------------------------------------------------
    # Image
    # --------------------------------------------------

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    image_np = np.asarray(
        image
    )

    transform = weights.transforms()

    input_tensor = (
        transform(
            image
        )
        .unsqueeze(0)
        .to(device)
    )

    print(
        f"Image: {IMAGE_PATH}"
    )

    print(
        f"Image size: {image.size}"
    )

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    print()
    print(
        "[1] Prediction"
    )

    with torch.no_grad():
        output = model(
            input_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1,
        )

    target = int(
        probabilities.argmax(
            dim=1
        ).item()
    )

    class_name = weights.meta[
        "categories"
    ][target]

    confidence = float(
        probabilities[
            0,
            target,
        ].item()
    )

    print(
        f"    Class: {class_name}"
    )

    print(
        f"    Class ID: {target}"
    )

    print(
        f"    Confidence: {confidence:.4f}"
    )

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------

    print()
    print(
        "[2] Grad-CAM"
    )

    engine = create_default_engine()

    target_layer = (
        model.layer4[-1].conv2
    )

    gradcam_result = engine.explain(
        method="grad_cam",
        model=model,
        input_data=input_tensor,
        target=target,
        target_layer=target_layer,
    )

    print(
        f"    Runtime: "
        f"{gradcam_result.runtime_ms:.2f} ms"
    )

    print(
        f"    Attribution: "
        f"{gradcam_result.attribution.shape}"
    )

    # --------------------------------------------------
    # LIME
    # --------------------------------------------------

    print()
    print(
        "[3] LIME Image"
    )

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

    lime_explainer = LimeImageExplainer(
        preprocess=preprocess,
        num_samples=500,
        hide_color=0,
    )

    lime_start = time.perf_counter()

    lime_result = (
        lime_explainer.explain(
            model=model,
            input_data=image_np,
            target=target,
        )
    )

    # Use the method's own runtime measurement.
    lime_runtime_ms = (
        time.perf_counter()
        - lime_start
    ) * 1000.0

    print(
        f"    Runtime: "
        f"{lime_result.runtime_ms:.2f} ms"
    )

    print(
        f"    Wall runtime: "
        f"{lime_runtime_ms:.2f} ms"
    )

    print(
        f"    Attribution: "
        f"{lime_result.attribution.shape}"
    )

    print(
        f"    Segments: "
        f"{lime_result.metadata['num_segments']}"
    )

    # --------------------------------------------------
    # Save raw attribution maps
    # --------------------------------------------------

    np.save(
        OUTPUT_DIR
        / "gradcam_attribution.npy",
        gradcam_result.attribution,
    )

    np.save(
        OUTPUT_DIR
        / "lime_attribution.npy",
        lime_result.attribution,
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    print()
    print(
        "[4] Saving attribution comparison"
    )

    gradcam_positive = (
        positive_attribution(
            gradcam_result.attribution
        )
    )

    lime_positive = (
        positive_attribution(
            lime_result.attribution
        )
    )

    gradcam_overlay = overlay_heatmap(
        image=image,
        heatmap=gradcam_positive,
        alpha=0.45,
    )

    lime_overlay = overlay_heatmap(
        image=image,
        heatmap=lime_positive,
        alpha=0.45,
    )

    comparison_image_path = (
        OUTPUT_DIR
        / "attribution_comparison.png"
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    axes[0].imshow(image)
    axes[0].set_title(
        "Original"
    )
    axes[0].axis("off")

    axes[1].imshow(
        gradcam_overlay
    )
    axes[1].set_title(
        f"Grad-CAM\n"
        f"{gradcam_result.runtime_ms:.1f} ms"
    )
    axes[1].axis("off")

    axes[2].imshow(
        lime_overlay
    )
    axes[2].set_title(
        f"LIME\n"
        f"{lime_result.runtime_ms:.1f} ms"
    )
    axes[2].axis("off")

    plt.tight_layout()

    plt.savefig(
        comparison_image_path,
        dpi=150,
    )

    plt.close()

    print(
        f"    Saved: "
        f"{comparison_image_path}"
    )

    # --------------------------------------------------
    # Faithfulness
    # --------------------------------------------------

    print()
    print(
        "[5] Faithfulness evaluation"
    )

    steps = 20
    baseline_value = 0.0

    # Grad-CAM deletion
    gradcam_deletion = compute_deletion(
        model=model,
        input_tensor=input_tensor,
        attribution=gradcam_result.attribution,
        target=target,
        steps=steps,
        baseline_value=baseline_value,
    )

    # Grad-CAM insertion
    gradcam_insertion = compute_insertion(
        model=model,
        input_tensor=input_tensor,
        attribution=gradcam_result.attribution,
        target=target,
        steps=steps,
        baseline_value=baseline_value,
    )

    # LIME deletion
    lime_deletion = compute_deletion(
        model=model,
        input_tensor=input_tensor,
        attribution=lime_result.attribution,
        target=target,
        steps=steps,
        baseline_value=baseline_value,
    )

    # LIME insertion
    lime_insertion = compute_insertion(
        model=model,
        input_tensor=input_tensor,
        attribution=lime_result.attribution,
        target=target,
        steps=steps,
        baseline_value=baseline_value,
    )

    print(
        f"    Grad-CAM deletion AUC: "
        f"{gradcam_deletion.auc:.4f}"
    )

    print(
        f"    Grad-CAM insertion AUC: "
        f"{gradcam_insertion.auc:.4f}"
    )

    print(
        f"    LIME deletion AUC: "
        f"{lime_deletion.auc:.4f}"
    )

    print(
        f"    LIME insertion AUC: "
        f"{lime_insertion.auc:.4f}"
    )

    # --------------------------------------------------
    # Faithfulness plot
    # --------------------------------------------------

    faithfulness_path = (
        OUTPUT_DIR
        / "faithfulness_comparison.png"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    axes[0].plot(
        gradcam_deletion.fractions,
        gradcam_deletion.confidences,
        label=(
            f"Grad-CAM "
            f"(AUC={gradcam_deletion.auc:.4f})"
        ),
    )

    axes[0].plot(
        lime_deletion.fractions,
        lime_deletion.confidences,
        label=(
            f"LIME "
            f"(AUC={lime_deletion.auc:.4f})"
        ),
    )

    axes[0].set_title(
        "Deletion"
    )

    axes[0].set_xlabel(
        "Fraction removed"
    )

    axes[0].set_ylabel(
        "Target confidence"
    )

    axes[0].legend()
    axes[0].grid(
        alpha=0.3
    )

    axes[1].plot(
        gradcam_insertion.fractions,
        gradcam_insertion.confidences,
        label=(
            f"Grad-CAM "
            f"(AUC={gradcam_insertion.auc:.4f})"
        ),
    )

    axes[1].plot(
        lime_insertion.fractions,
        lime_insertion.confidences,
        label=(
            f"LIME "
            f"(AUC={lime_insertion.auc:.4f})"
        ),
    )

    axes[1].set_title(
        "Insertion"
    )

    axes[1].set_xlabel(
        "Fraction restored"
    )

    axes[1].set_ylabel(
        "Target confidence"
    )

    axes[1].legend()
    axes[1].grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        faithfulness_path,
        dpi=150,
    )

    plt.close()

    print(
        f"    Saved: "
        f"{faithfulness_path}"
    )

    # --------------------------------------------------
    # JSON result
    # --------------------------------------------------

    print()
    print(
        "[6] Saving structured result"
    )

    result = {
        "experiment": {
            "name": "image_xai_comparison",
            "image": str(
                IMAGE_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "model": "resnet18",
            "device": str(device),
        },
        "prediction": {
            "class_id": target,
            "class_name": class_name,
            "confidence": confidence,
        },
        "methods": {
            "grad_cam": {
                "runtime_ms": (
                    gradcam_result.runtime_ms
                ),
                "attribution_shape": list(
                    gradcam_result.attribution.shape
                ),
            },
            "lime": {
                "runtime_ms": (
                    lime_result.runtime_ms
                ),
                "num_samples": 500,
                "num_segments": (
                    lime_result.metadata[
                        "num_segments"
                    ]
                ),
                "attribution_shape": list(
                    lime_result.attribution.shape
                ),
            },
        },
        "faithfulness": {
            "steps": steps,
            "baseline_value": baseline_value,
            "grad_cam": {
                "deletion_auc": (
                    gradcam_deletion.auc
                ),
                "insertion_auc": (
                    gradcam_insertion.auc
                ),
            },
            "lime": {
                "deletion_auc": (
                    lime_deletion.auc
                ),
                "insertion_auc": (
                    lime_insertion.auc
                ),
            },
        },
        "artifacts": {
            "attribution_comparison": (
                "attribution_comparison.png"
            ),
            "faithfulness_comparison": (
                "faithfulness_comparison.png"
            ),
            "gradcam_attribution": (
                "gradcam_attribution.npy"
            ),
            "lime_attribution": (
                "lime_attribution.npy"
            ),
        },
    }

    result_path = (
        OUTPUT_DIR
        / "comparison.json"
    )

    save_json(
        result_path,
        result,
    )

    print(
        f"    Saved: {result_path}"
    )

    engine.close()

    print()
    print("=" * 70)
    print(
        "IMAGE XAI COMPARISON COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()