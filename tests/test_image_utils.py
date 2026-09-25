import pytest
import torch

from uav_xai.utils import (
    validate_image_tensor,
    tensor_to_numpy_image,
)


def test_validate_image_tensor():
    image = torch.rand(
        1,
        3,
        224,
        224,
    )

    result = validate_image_tensor(
        image
    )

    assert result.shape == (
        1,
        3,
        224,
        224,
    )

    assert torch.is_floating_point(
        result
    )


def test_validate_image_tensor_rejects_wrong_shape():
    image = torch.rand(
        3,
        224,
        224,
    )

    with pytest.raises(ValueError):
        validate_image_tensor(
            image
        )


def test_tensor_to_numpy_image():
    image = torch.rand(
        1,
        3,
        224,
        224,
    )

    result = tensor_to_numpy_image(
        image
    )

    assert result.shape == (
        224,
        224,
        3,
    )