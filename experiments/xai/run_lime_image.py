from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)

from uav_xai.methods import LimeImageExplainer
from uav_xai.utils import save_image


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
    print("LIME IMAGE EXPERIMENT")
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
    # Original image
    # --------------------------------------------------

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    image_np = np.asarray(
        image
    )

    print(
        f"Image: {IMAGE_PATH}"
    )

    print(
        f"Image size: {image.size}"
    )

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Original prediction
    # --------------------------------------------------

    input_tensor = transform(
        image
    ).unsqueeze(0).to(device)

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
    # LIME
    # --------------------------------------------------

    explainer = LimeImageExplainer(
        preprocess=preprocess,
        num_samples=500,
        hide_color=0,
    )

    print()
    print(
        "Running LIME..."
    )

    result = explainer.explain(
        model=model,
        input_data=image_np,
        target=target,
    )

    print()
    print(
        f"LIME runtime: "
        f"{result.runtime_ms:.2f} ms"
    )

    print(
        f"Attribution shape: "
        f"{result.attribution.shape}"
    )

    print(
        f"Segments: "
        f"{result.metadata['num_segments']}"
    )

    # --------------------------------------------------
    # Normalize attribution for visualization
    # --------------------------------------------------

    attribution = (
        result.attribution
    )

    abs_max = np.max(
        np.abs(attribution)
    )

    if abs_max > 0:
        normalized = (
            attribution / abs_max
        )

    else:
        normalized = np.zeros_like(
            attribution
        )

    # Positive contribution only.
    positive = np.maximum(
        normalized,
        0.0,
    )

    # --------------------------------------------------
    # Save raw attribution map
    # --------------------------------------------------

    raw_path = (
        OUTPUT_DIR
        / "lime_image_attribution.npy"
    )

    np.save(
        raw_path,
        attribution,
    )

    print(
        f"Saved attribution: "
        f"{raw_path}"
    )

    # --------------------------------------------------
    # Save LIME heatmap
    # --------------------------------------------------

    heatmap_path = (
        OUTPUT_DIR
        / "lime_image_heatmap.png"
    )

    from uav_xai.utils import (
        overlay_heatmap,
    )

    heatmap_image = overlay_heatmap(
        image=image,
        heatmap=positive,
        alpha=0.45,
    )

    save_image(
        heatmap_image,
        heatmap_path,
    )

    print(
        f"Saved heatmap: "
        f"{heatmap_path}"
    )

    # --------------------------------------------------
    # Save superpixel visualization
    # --------------------------------------------------

    segments_path = (
        OUTPUT_DIR
        / "lime_image_segments.png"
    )

    segments = np.asarray(
        result.metadata["segments"]
    )

    segments_image = (
        LimeImageExplainer.visualize_segments(
            image_np,
            segments,
        )
    )

    # Use the actual LIME segmentation.
    # The segmentation is stored inside the
    # LIME explanation object, so we perform
    # a second explanation only if needed.
    #
    # For now, use the attribution map as
    # the segmentation visualization source
    # only as a placeholder.

    from PIL import Image as PILImage

    segments_uint8 = (
        np.clip(
            positive * 255.0,
            0,
            255,
        )
        .astype(np.uint8)
    )

    PILImage.fromarray(
        (segments_image * 255).clip(
            0,
            255,
        ).astype(np.uint8)
    ).save(
        segments_path
    )

    print(
        f"Saved segments: "
        f"{segments_path}"
    )

    print()
    print(
        "LIME experiment complete."
    )


if __name__ == "__main__":
    main()