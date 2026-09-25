import pytest

from modules.obstacle_detector import ObstacleDetector


def test_detector_no_obstacle():
    detector = ObstacleDetector()

    state = {
        "position": {
            "x": 10.0,
            "y": 5.0,
            "z": -1.0,
        },
        "velocity": {
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "orientation": {
            "yaw": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
        },
        "sensors": {
            "obstacle_distance": 100.0,
            "obstacle_density": 0.0,
            "battery": 100.0,
            "collision": False,
        },
    }

    result = detector.detect(state)

    assert result is None


def test_detector_critical_obstacle():
    detector = ObstacleDetector()

    state = {
        "position": {
            "x": 10.0,
            "y": 5.0,
            "z": -1.0,
        },
        "velocity": {
            "vx": 2.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "orientation": {
            "yaw": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
        },
        "sensors": {
            "obstacle_distance": 2.0,
            "obstacle_density": 0.2,
            "battery": 80.0,
            "collision": False,
            "object_name": "wall",
        },
    }

    result = detector.detect(state)

    assert result is not None
    assert result["type"] == "collision"
    assert result["distance"] == pytest.approx(2.0)
    assert result["object"] == "стена"


def test_detector_collision():
    detector = ObstacleDetector()

    state = {
        "position": {
            "x": 10.0,
            "y": 5.0,
            "z": -1.0,
        },
        "velocity": {
            "vx": 1.0,
            "vy": 0.0,
            "vz": 0.0,
        },
        "orientation": {
            "yaw": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
        },
        "sensors": {
            "obstacle_distance": 0.0,
            "obstacle_density": 1.0,
            "battery": 80.0,
            "collision": True,
            "object_name": "building",
        },
    }

    result = detector.detect(state)

    assert result is not None
    assert result["type"] == "collision"
    assert result["object"] == "здание"