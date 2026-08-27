from pathlib import Path
from typing import Any

import yaml


def load_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """
    Load experiment configuration from YAML.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Config root must be a YAML mapping."
        )

    return config