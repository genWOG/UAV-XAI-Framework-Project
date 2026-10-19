from types import SimpleNamespace

import numpy as np
import pytest

from uav_xai.methods.yolo_fast_scorer import YoloDetectionScorer


class FakeBoxes:
    def __init__(self, xyxy, classes, confidences):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.cls = np.asarray(classes, dtype=np.float32)
        self.conf = np.asarray(confidences, dtype=np.float32)

    def __len__(self):
        return len(self.cls)


class FakeYOLO:
    def __init__(self, fast_confidence=0.8, default_confidence=0.8):
        self.fast_confidence = fast_confidence
        self.default_confidence = default_confidence
        self.calls = []

    def predict(self, source, conf, verbose, **kwargs):
        nms_free = kwargs.get("nms", None) is False
        self.calls.append((len(source), nms_free))
        confidence = self.fast_confidence if nms_free else self.default_confidence
        return [
            SimpleNamespace(
                boxes=FakeBoxes([[1, 1, 5, 5]], [1], [confidence])
            )
            for _ in source
        ]


class FakeYOLOWithoutNMSFlag:
    def __init__(self):
        self.calls = 0

    def predict(self, source, conf, verbose):
        self.calls += 1
        return [
            SimpleNamespace(
                boxes=FakeBoxes([[1, 1, 5, 5]], [1], [0.7])
            )
            for _ in source
        ]


def _image():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_uses_nms_free_after_baseline_score_matches():
    model = FakeYOLO(fast_confidence=0.8, default_confidence=0.8)
    scorer = YoloDetectionScorer(model, [1, 1, 5, 5], 1)

    first = scorer([_image()])
    assert first[0, 1] == pytest.approx(0.8)
    assert scorer.backend == "ultralytics_predict_nms_free"
    assert scorer.max_score_difference == pytest.approx(0.0)

    second = scorer([_image(), _image()])
    assert second[:, 1] == pytest.approx([0.8, 0.8])
    assert model.calls == [(1, False), (1, True), (2, True)]


def test_falls_back_when_nms_free_score_diverges():
    model = FakeYOLO(fast_confidence=0.5, default_confidence=0.8)
    scorer = YoloDetectionScorer(model, [1, 1, 5, 5], 1, parity_tolerance=0.05)

    with pytest.warns(RuntimeWarning, match="differs"):
        first = scorer([_image()])
    assert first[0, 1] == pytest.approx(0.8)
    assert scorer.backend == "ultralytics_predict_fallback_parity"

    second = scorer([_image()])
    assert second[0, 1] == pytest.approx(0.8)
    assert model.calls == [(1, False), (1, True), (1, False)]


def test_falls_back_when_nms_argument_is_not_supported():
    model = FakeYOLOWithoutNMSFlag()
    scorer = YoloDetectionScorer(model, [1, 1, 5, 5], 1)

    with pytest.warns(RuntimeWarning, match="unavailable"):
        first = scorer([_image()])
    assert first[0, 1] == pytest.approx(0.7)
    assert scorer.backend == "ultralytics_predict_fallback_unsupported"
    assert scorer([_image()])[0, 1] == pytest.approx(0.7)
    assert model.calls == 2
