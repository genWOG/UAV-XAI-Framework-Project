import numpy as np

from sklearn.ensemble import RandomForestClassifier

from uav_xai import (
    create_default_engine,
)


def test_lime_tabular_explanation():
    rng = np.random.default_rng(42)

    X = rng.normal(
        size=(200, 8)
    ).astype(np.float32)

    y = (
        X[:, 0] + X[:, 1] - X[:, 2]
        > 0
    ).astype(int)

    feature_names = [
        "altitude",
        "speed",
        "obstacle_dist",
        "battery",
        "obstacle_density",
        "yaw",
        "roll",
        "pitch",
    ]

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
    )

    model.fit(
        X,
        y,
    )

    instance = X[0]

    engine = create_default_engine()

    result = engine.explain(
        method="lime",
        model=model,
        input_data=instance,
        target=int(
            model.predict(instance.reshape(1, -1))[0]
        ),
        training_data=X,
        feature_names=feature_names,
        class_names=[
            "stop",
            "continue",
        ],
        num_samples=200,
    )

    assert result.method == "lime"

    assert result.input_type == "tabular"

    assert result.attribution.shape == (
        8,
    )

    assert result.feature_names == (
        feature_names
    )

    assert result.runtime_ms is not None

    assert result.confidence is not None

    assert result.target in (
        0,
        1,
    )

    assert np.isfinite(
        result.attribution
    ).all()

    engine.close()