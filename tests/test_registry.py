import pytest

from uav_xai.core.base import Explainer
from uav_xai.core.registry import (
    ExplainerRegistry,
)
from uav_xai.core.result import ExplanationResult
from uav_xai.core.engine import XAIEngine

class DummyExplainer(Explainer):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def explain(
        self,
        model,
        input_data,
        target=None,
    ) -> ExplanationResult:
        raise NotImplementedError


def test_registry_resolves_by_input_type_and_method():
    registry = ExplainerRegistry()

    registry.register(
        input_type="image",
        method="lime",
        explainer_cls=DummyExplainer,
    )

    registry.register(
        input_type="tabular",
        method="lime",
        explainer_cls=DummyExplainer,
    )

    assert (
        registry.get(
            "image",
            "lime",
        )
        is DummyExplainer
    )

    assert (
        registry.get(
            "tabular",
            "lime",
        )
        is DummyExplainer
    )


def test_registry_stores_cache_metadata():
    registry = ExplainerRegistry()

    registry.register(
        input_type="image",
        method="grad_cam",
        explainer_cls=DummyExplainer,
        cache_instance=True,
    )

    spec = registry.get_spec(
        "image",
        "grad_cam",
    )

    assert (
        spec.explainer_cls
        is DummyExplainer
    )

    assert spec.cache_instance is True


def test_registry_rejects_duplicate_registration():
    registry = ExplainerRegistry()

    registry.register(
        input_type="image",
        method="lime",
        explainer_cls=DummyExplainer,
    )

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register(
            input_type="image",
            method="lime",
            explainer_cls=DummyExplainer,
        )


def test_registry_rejects_non_explainer():
    registry = ExplainerRegistry()

    class NotAnExplainer:
        pass

    with pytest.raises(
        TypeError,
        match="must inherit from Explainer",
    ):
        registry.register(
            input_type="image",
            method="test",
            explainer_cls=NotAnExplainer,
        )

def test_registry_builds_explainer_kwargs():
    registry = ExplainerRegistry()

    def build_kwargs(context):
        return {
            "value": context["value"]
        }

    registry.register(
        input_type="image",
        method="dummy",
        explainer_cls=DummyExplainer,
        build_kwargs=build_kwargs,
    )

    kwargs = registry.build_explainer_kwargs(
        input_type="image",
        method="dummy",
        context={
            "value": 42
        },
    )

    assert kwargs == {
        "value": 42
    }

def test_engine_uses_registry_builder():
    registry = ExplainerRegistry()

    class BuilderExplainer(DummyExplainer):
        def __init__(self, value):
            self.value = value

    def build_kwargs(context):
        return {
            "value": context["value"]
        }

    registry.register(
        input_type="image",
        method="dummy",
        explainer_cls=BuilderExplainer,
        build_kwargs=build_kwargs,
    )

    engine = XAIEngine(
        registry=registry
    )

    instance = engine._get_instance(
        input_type="image",
        method="dummy",
        context={
            "value": 123
        },
    )

    assert instance.value == 123

def test_registry_contains_image_shap():
    from uav_xai import create_default_engine

    engine = create_default_engine()

    assert (
        "shap"
        in engine.list_methods(
            "image"
        )
    )

    engine.close()

def test_registry_contains_image_lrp():
    from uav_xai import create_default_engine

    engine = create_default_engine()

    assert (
        "lrp"
        in engine.list_methods(
            "image"
        )
    )

    engine.close()