from pathlib import Path

import yaml

from uav_xai.core.benchmark import (
    BenchmarkRunner,
)


def test_benchmark_config_is_valid(tmp_path):

    config = {
        "experiment": {
            "name": "test_benchmark",
        },
        "benchmark": {
            "runs": 2,
        },
        "input": {
            "type": "image",
            "paths": [
                "test.jpg",
            ],
        },
        "model": {
            "name": "resnet18",
        },
        "xai": {
            "methods": [
                "grad_cam",
                "lime",
                "shap",
                "lrp",
            ],
        },
        "metrics": {
            "enabled": True,
        },
        "output": {
            "root": "results",
        },
    }

    config_path = (
        tmp_path
        / "benchmark.yaml"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
        )

    runner = BenchmarkRunner(
        config_path
    )

    assert runner.config[
        "benchmark"
    ]["runs"] == 2

    assert runner._get_methods() == [
        "grad_cam",
        "lime",
        "shap",
        "lrp",
    ]

    assert runner._get_paths() == [
        Path("test.jpg")
    ]