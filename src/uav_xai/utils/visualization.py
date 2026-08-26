from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap on top of the original image.

    Parameters
    ----------
    image:
        Original PIL image.
    heatmap:
        2D numpy array with values in [0, 1].
    alpha:
        Transparency of the heatmap.

    Returns
    -------
    PIL.Image.Image
        Resulting visualization.
    """

    if heatmap.ndim != 2:
        raise ValueError(
            f"Expected 2D heatmap, got shape {heatmap.shape}"
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    # Normalize heatmap to [0, 1].
    heatmap = np.asarray(heatmap, dtype=np.float32)

    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()

    if heatmap_max > heatmap_min:
        heatmap = (
            heatmap - heatmap_min
        ) / (heatmap_max - heatmap_min)

    # Resize heatmap to original image dimensions.
    heatmap_image = Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    )

    heatmap_image = heatmap_image.resize(
        image.size,
        Image.Resampling.BILINEAR,
    )

    # Apply matplotlib colormap.
    cmap = plt.get_cmap("jet")

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32,
    ) / 255.0

    colored_heatmap = cmap(heatmap_array)[:, :, :3]

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(np.uint8)

    colored_heatmap_image = Image.fromarray(
        colored_heatmap
    )

    # Blend original image and heatmap.
    original_rgb = image.convert("RGB")

    result = Image.blend(
        original_rgb,
        colored_heatmap_image,
        alpha,
    )

    return result


def save_image(
    image: Image.Image,
    path: str | Path,
) -> None:
    """
    Save a PIL image to disk.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(path)

def plot_faithfulness_curves(
    deletion_fractions,
    deletion_confidences,
    deletion_auc,
    insertion_fractions,
    insertion_confidences,
    insertion_auc,
    output_path: str | Path,
) -> None:
    """
    Plot deletion and insertion faithfulness curves.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    # Deletion curve.
    axes[0].plot(
        deletion_fractions,
        deletion_confidences,
        marker="o",
    )

    axes[0].set_title(
        f"Deletion\nAUC = {deletion_auc:.4f}"
    )

    axes[0].set_xlabel(
        "Fraction of pixels removed"
    )

    axes[0].set_ylabel(
        "Target confidence"
    )

    axes[0].set_xlim(
        0.0,
        1.0,
    )

    axes[0].set_ylim(
        0.0,
        1.0,
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    # Insertion curve.
    axes[1].plot(
        insertion_fractions,
        insertion_confidences,
        marker="o",
    )

    axes[1].set_title(
        f"Insertion\nAUC = {insertion_auc:.4f}"
    )

    axes[1].set_xlabel(
        "Fraction of pixels inserted"
    )

    axes[1].set_ylabel(
        "Target confidence"
    )

    axes[1].set_xlim(
        0.0,
        1.0,
    )

    axes[1].set_ylim(
        0.0,
        1.0,
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)