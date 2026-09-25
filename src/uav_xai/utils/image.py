from typing import Any

import numpy as np
import torch


def validate_image_tensor(
    input_data: Any,
) -> torch.Tensor:
    """
    Validate and normalize an image tensor.

    Expected shape:
        [1, C, H, W]

    Accepted channel counts:
        1 or 3.
    """

    if not isinstance(
        input_data,
        torch.Tensor,
    ):
        raise TypeError(
            "Image input must be a torch.Tensor."
        )

    if input_data.ndim != 4:
        raise ValueError(
            "Image tensor must have shape "
            "[batch, channels, height, width]."
        )

    if input_data.shape[0] != 1:
        raise ValueError(
            "Only batch size 1 is supported."
        )

    if input_data.shape[1] not in (1, 3):
        raise ValueError(
            "Image tensor must have 1 or 3 channels."
        )

    if input_data.shape[2] < 2 or input_data.shape[3] < 2:
        raise ValueError(
            "Image spatial dimensions must be at least 2x2."
        )

    if not torch.is_floating_point(input_data):
        input_data = input_data.float()

    if not torch.isfinite(input_data).all():
        raise ValueError(
            "Image tensor contains NaN or Inf values."
        )

    return input_data


def tensor_to_numpy_image(
    input_data: torch.Tensor,
) -> np.ndarray:
    """
    Convert [1, C, H, W] tensor to [H, W, C] numpy array.
    """

    input_data = validate_image_tensor(
        input_data
    )

    image = (
        input_data[0]
        .detach()
        .cpu()
        .numpy()
    )

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    return image