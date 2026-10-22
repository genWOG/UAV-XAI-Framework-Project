import numpy as np
import torch

from uav_xai.core.target import XAITarget
from uav_xai.methods import ShapImageExplainer


class FakeBoxes:
    def __init__(self, confidence: float, bbox=(2.0, 2.0, 14.0, 14.0)) -> None:
        self.xyxy = torch.tensor([bbox], dtype=torch.float32)
        self.cls = torch.tensor([1.0])
        self.conf = torch.tensor([confidence])

    def __len__(self) -> int:
        return int(self.cls.numel())


class FakeResult:
    def __init__(self, confidence: float, bbox=(2.0, 2.0, 14.0, 14.0)) -> None:
        self.boxes = FakeBoxes(confidence, bbox)


class FakeYOLO:
    def __init__(self, bbox=(2.0, 2.0, 14.0, 14.0)):
        self.batch_sizes = []
        self.bbox = bbox

    def predict(self, source, conf=0.001, verbose=False):
        del conf, verbose
        self.batch_sizes.append(len(source))
        return [
            FakeResult(float(np.asarray(image).mean() / 255.0), self.bbox)
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
    model = FakeYOLO()
    explainer = ShapImageExplainer(
        preprocess=None,
        nsamples=12,
        num_segments=4,
        baseline_mode="segment_mean",
        target_iou_threshold=0.5,
        prediction_batch_size=3,
    )

    result = explainer.explain(
        model=model,
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
    assert result.metadata["target_score_mode"] == "confidence_iou"
    assert result.metadata["prediction_batch_size"] == 3
    assert model.batch_sizes
    assert max(model.batch_sizes) <= 3



def test_shap_yolo_uses_continuous_confidence_iou_score_below_threshold():
    model = FakeYOLO(bbox=(7.0, 2.0, 19.0, 14.0))
    target = XAITarget(
        task="detection",
        class_id=1,
        score=0.8,
        detection_index=0,
        bbox=[2.0, 2.0, 14.0, 14.0],
    )
    image = np.full((16, 16, 3), 128, dtype=np.uint8)

    continuous_explainer = ShapImageExplainer(
        preprocess=None,
        nsamples=4,
        num_segments=4,
        target_iou_threshold=0.5,
        target_score_mode="confidence_iou",
    )
    continuous = continuous_explainer._build_detection_predict_fn(
        model, target
    )(np.expand_dims(image, axis=0))

    # IoU is 7/17 (< 0.5); raw class confidence is image mean / 255.
    expected = (128.0 / 255.0) * (7.0 / 17.0)
    assert np.isclose(continuous[0, 1], expected, atol=1e-6)
    assert np.isclose(continuous[0].sum(), 1.0, atol=1e-6)

    thresholded = ShapImageExplainer(
        preprocess=None,
        nsamples=4,
        num_segments=4,
        target_iou_threshold=0.5,
        target_score_mode="thresholded_confidence",
    )._build_detection_predict_fn(model, target)(np.expand_dims(image, axis=0))
    assert thresholded[0, 1] == 0.0
