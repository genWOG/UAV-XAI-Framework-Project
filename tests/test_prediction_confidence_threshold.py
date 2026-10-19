from types import SimpleNamespace

import pytest
import torch

from uav_xai.core.predictor import ModelPredictor


class FakeBoxes:
    def __init__(self):
        self.xyxy = torch.tensor([[1.0, 2.0, 10.0, 12.0]])
        self.cls = torch.tensor([8.0])
        self.conf = torch.tensor([0.75])

    def __len__(self):
        return 1


class FakeYOLO:
    def __init__(self):
        self.kwargs = None

    def predict(self, source, verbose=False, **kwargs):
        self.kwargs = {"source": source, "verbose": verbose, **kwargs}
        return [SimpleNamespace(boxes=FakeBoxes(), names={8: "boat"})]


def test_default_predictor_does_not_override_ultralytics_confidence():
    model = FakeYOLO()
    result = ModelPredictor(model, task="detection").predict(torch.zeros(1, 3, 16, 16))

    assert result.confidence == pytest.approx(0.75)
    assert "conf" not in model.kwargs


def test_configured_confidence_threshold_is_forwarded():
    model = FakeYOLO()
    result = ModelPredictor(
        model, task="detection", confidence_threshold=0.001
    ).predict(torch.zeros(1, 3, 16, 16))

    assert result.confidence == pytest.approx(0.75)
    assert model.kwargs["conf"] == pytest.approx(0.001)


def test_confidence_threshold_is_validated():
    with pytest.raises(ValueError, match="confidence_threshold"):
        ModelPredictor(FakeYOLO(), task="detection", confidence_threshold=1.1)
