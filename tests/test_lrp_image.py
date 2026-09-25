import numpy as np
import torch
import torch.nn as nn

from uav_xai.methods import LrpImageExplainer


class TinyImageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            3,
            4,
            kernel_size=3,
            padding=1,
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc = nn.Linear(
            4,
            2,
        )

    def forward(self, x):
        x = torch.relu(
            self.conv(x)
        )

        x = self.pool(x)

        x = x.flatten(1)

        return self.fc(x)


def preprocess(
    image: np.ndarray,
) -> torch.Tensor:
    tensor = (
        torch.from_numpy(
            image
        ).float()
        / 255.0
    )

    return tensor.permute(
        2,
        0,
        1,
    )


def test_lrp_image_explanation():

    rng = np.random.default_rng(
        42
    )

    image = (
        rng.random(
            (16, 16, 3)
        ) * 255
    ).astype(
        np.uint8
    )

    model = TinyImageModel()
    model.eval()

    explainer = LrpImageExplainer(
        preprocess=preprocess
    )

    result = explainer.explain(
        model=model,
        input_data=image,
    )

    assert result.method == "lrp"
    assert result.input_type == "image"

    assert result.attribution.shape == (
        16,
        16,
    )

    assert np.isfinite(
        result.attribution
    ).all()

    assert result.target in (
        0,
        1,
    )

    assert result.confidence is not None
    assert result.runtime_ms is not None