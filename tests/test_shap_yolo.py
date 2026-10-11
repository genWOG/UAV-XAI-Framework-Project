import numpy as np
import torch

from uav_xai.core.target import XAITarget
from uav_xai.methods import ShapImageExplainer


class FakeBoxes:
    def __init__(self, confidence: float) -> None:
        self.xyxy = torch.tensor([[2.0, 2.0, 14.0, 14.0]])
        self.cls = torch.tensor([1.0])
        self.conf = torch.tensor([confidence])

    def __len__(self) -> int:
        return int(self.cls.numel())


class FakeResult:
    def __init__(self, confidence: float) -> None:
        self.boxes = FakeBoxes(confidence)


class FakeYOLO:
    def predict(self, source, conf=0.001, verbose=False):
        del conf, verbose
        return [
            FakeResult(float(np.asarray(image).mean() / 255.0))
            for image in source
        ]


def test_shap_image_explanation_for_yolo_detection_target():
    rng = np.random.default_rng(123)
    image = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
    target = XAITarget(
        task="detection",
        class_id=1,
        score=0.8,
        detection_index=0,
        bbox=[2.0, 2.0, 14.0, 14.0],
    )
    explainer = ShapImageExplainer(
        preprocess=None,
        nsamples=12,
        num_segments=4,
        baseline_mode="segment_mean",
        target_iou_threshold=0.5,
    )

    result = explainer.explain(
        model=FakeYOLO(),
        input_data=image,
        target=target,
    )

    assert result.method == "shap"
    assert result.target == 1
    assert result.attribution.shape == image.shape[:2]
    assert np.isfinite(result.attribution).all()
    assert result.metadata["target_task"] == "detection"
    assert result.metadata["target_class_id"] == 1
    assert result.metadata["target_iou_threshold"] == 0.5
    assert result.metadata["baseline_mode"] == "segment_mean"
