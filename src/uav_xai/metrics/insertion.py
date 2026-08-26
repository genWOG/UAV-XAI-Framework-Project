from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InsertionResult:
    """
    Result of insertion faithfulness evaluation.
    """

    fractions: np.ndarray
    confidences: np.ndarray
    auc: float


def _prepare_attribution(
    attribution: np.ndarray,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """
    Resize attribution map to input image resolution.
    """

    attribution_tensor = torch.as_tensor(
        attribution,
        dtype=torch.float32,
    )

    if attribution_tensor.ndim != 2:
        raise ValueError(
            "Attribution must be a 2D array."
        )

    attribution_tensor = attribution_tensor.unsqueeze(
        0
    ).unsqueeze(0)

    attribution_tensor = F.interpolate(
        attribution_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    return attribution_tensor[0, 0]


def _get_confidence(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target: int,
) -> float:
    """
    Get model confidence for target class.
    """

    with torch.no_grad():
        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        return float(
            probabilities[0, target].item()
        )


def compute_insertion(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    attribution: np.ndarray,
    target: int,
    steps: int = 20,
    baseline_value: float = 0.0,
) -> InsertionResult:
    """
    Evaluate explanation faithfulness using insertion.

    The input starts from a baseline image and progressively
    restores the pixels with the highest attribution.

    A faithful explanation should cause the target confidence
    to increase quickly.

    Parameters
    ----------
    model:
        Model being explained.

    input_tensor:
        Original normalized input tensor with shape
        [1, C, H, W].

    attribution:
        2D attribution map.

    target:
        Target class.

    steps:
        Number of insertion steps.

    baseline_value:
        Initial value for all pixels.

    Returns
    -------
    InsertionResult
    """

    if input_tensor.ndim != 4:
        raise ValueError(
            "input_tensor must have shape [B, C, H, W]."
        )

    if input_tensor.shape[0] != 1:
        raise ValueError(
            "Only batch size 1 is supported."
        )

    if steps < 1:
        raise ValueError(
            "steps must be >= 1."
        )

    model.eval()

    _, channels, height, width = input_tensor.shape

    attribution_tensor = _prepare_attribution(
        attribution,
        height,
        width,
    )

    scores = attribution_tensor.flatten()

    sorted_indices = torch.argsort(
        scores,
        descending=True,
    )

    total_pixels = height * width

    fractions = np.linspace(
        0.0,
        1.0,
        steps + 1,
    )

    confidences: list[float] = []

    for fraction in fractions:

        num_pixels = int(
            round(
                fraction * total_pixels
            )
        )

        # Start from baseline.
        modified = torch.full_like(
            input_tensor,
            fill_value=baseline_value,
        )

        if num_pixels > 0:
            indices = sorted_indices[
                :num_pixels
            ]

            modified_flat = modified.view(
                channels,
                -1,
            )

            original_flat = input_tensor.view(
                channels,
                -1,
            )

            modified_flat[
                :,
                indices,
            ] = original_flat[
                :,
                indices,
            ]

        confidence = _get_confidence(
            model=model,
            input_tensor=modified,
            target=target,
        )

        confidences.append(
            confidence
        )

    confidences_array = np.asarray(
        confidences,
        dtype=np.float64,
    )

    auc = float(
        np.trapezoid(
            confidences_array,
            fractions,
        )
    )

    return InsertionResult(
        fractions=fractions,
        confidences=confidences_array,
        auc=auc,
    )