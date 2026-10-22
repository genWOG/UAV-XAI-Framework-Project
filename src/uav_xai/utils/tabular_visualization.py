"""
Visualization utilities for tabular XAI attributions.

Renders a horizontal bar chart of feature importances,
colored by sign (green = positive contribution,
red = negative contribution).
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt


def plot_tabular_attribution(
    attribution: np.ndarray,
    feature_names: list[str],
    output_path: str | Path,
    title: str = "Feature attribution",
    top_k: int | None = None,
    normalize: bool = False,
) -> None:
    """
    Save a horizontal bar chart of feature attributions.

    Parameters
    ----------
    attribution:
        1D array of length n_features.
    feature_names:
        Names of features (same length as attribution).
    output_path:
        Where to save the PNG.
    title:
        Plot title.
    top_k:
        If set, only show top-k features by absolute value.
    normalize:
        If True, scale attributions by their absolute sum.
    """

    attribution = np.asarray(
        attribution,
        dtype=np.float64,
    ).ravel()

    if attribution.ndim != 1:
        raise ValueError(
            f"attribution must be 1D, got shape {attribution.shape}"
        )

    if len(feature_names) != len(attribution):
        raise ValueError(
            f"feature_names length ({len(feature_names)}) "
            f"!= attribution length ({len(attribution)})"
        )

    if normalize:
        denom = np.abs(attribution).sum()
        if denom > 0:
            attribution = attribution / denom

    # Sort by absolute value (descending).
    order = np.argsort(np.abs(attribution))[::-1]

    if top_k is not None:
        order = order[:top_k]

    sorted_values = attribution[order]
    sorted_names = [feature_names[i] for i in order]

    # Plot.
    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.4 * len(sorted_names)))
    )

    colors = [
        "green" if v > 0 else "red"
        for v in sorted_values
    ]

    y_pos = np.arange(len(sorted_names))

    ax.barh(
        y_pos,
        sorted_values,
        color=colors,
        alpha=0.75,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Contribution to decision")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)