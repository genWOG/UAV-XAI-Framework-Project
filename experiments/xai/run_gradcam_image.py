from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)

from uav_xai import create_default_engine
from uav_xai.utils import (
    overlay_heatmap,
    save_image,
)


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
    / "gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("GRAD-CAM IMAGE EXPERIMENT")
    print("=" * 60)

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

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print(
        f"Image: {IMAGE_PATH}"
    )

    print(
        f"Image size: {image.size}"
    )

    transform = weights.transforms()

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

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

    confidence = float(
        probabilities[
            0,
            target,
        ].item()
    )

    class_name = weights.meta[
        "categories"
    ][target]

    print()
    print(
        f"Prediction: {class_name}"
    )

    print(
        f"Class ID: {target}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    # --------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------

    target_layer = (
        model.layer4[-1].conv2
    )

    engine = create_default_engine()

    print()
    print(
        "Running Grad-CAM..."
    )

    result = engine.explain(
        method="grad_cam",
        model=model,
        input_data=input_tensor,
        target=target,
        target_layer=target_layer,
    )

    print()
    print(
        f"Grad-CAM runtime: "
        f"{result.runtime_ms:.2f} ms"
    )

    print(
        f"Attribution shape: "
        f"{result.attribution.shape}"
    )

    # --------------------------------------------------
    # Save raw attribution
    # --------------------------------------------------

    attribution_path = (
        OUTPUT_DIR
        / "gradcam_attribution.npy"
    )

    np.save(
        attribution_path,
        result.attribution,
    )

    print(
        f"Saved attribution: "
        f"{attribution_path}"
    )

    # --------------------------------------------------
    # Save heatmap
    # --------------------------------------------------

    heatmap_path = (
        OUTPUT_DIR
        / "gradcam_heatmap.png"
    )

    visualization = overlay_heatmap(
        image=image,
        heatmap=result.attribution,
        alpha=0.45,
    )

    save_image(
        visualization,
        heatmap_path,
    )

    print(
        f"Saved heatmap: "
        f"{heatmap_path}"
    )

    engine.close()

    print()
    print(
        "Grad-CAM experiment complete."
    )


if __name__ == "__main__":
    main()