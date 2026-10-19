import numpy as np
import torch
import torch.nn as nn

from uav_xai.core.target import XAITarget
from uav_xai.core.target_builder import TargetBuilder
from uav_xai.methods import LrpImageExplainer


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class FakeYOLO:
    def __init__(self):
        self.model = TinyDetector()


def test_lrp_image_explanation_for_detection_target(monkeypatch):
    def fake_build(model, input_data, target, context=None):
        del target, context
        return model.model(input_data)[0, 0]

    monkeypatch.setattr(TargetBuilder, "build", staticmethod(fake_build))
    rng = np.random.default_rng(42)
    image = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
    context = {"xai_input": torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0}
    target = XAITarget(
        task="detection",
        class_id=1,
        score=0.82,
        detection_index=0,
        bbox=[2.0, 2.0, 14.0, 14.0],
    )

    result = LrpImageExplainer(
        preprocess=None,
        context=context,
    ).explain(
        model=FakeYOLO(),
        input_data=image,
        target=target,
    )

    assert result.method == "lrp"
    assert result.target == 1
    assert result.attribution.shape == image.shape[:2]
    assert np.isfinite(result.attribution).all()
    assert result.metadata["target_task"] == "detection"
    assert result.metadata["target_class_id"] == 1
