"""
Experiment runner for tabular XAI experiments.

Analogous to `ExperimentRunner` (image-based), but
operates on tabular data with LIME / SHAP.

Supports two data sources:
    - "synthetic":   generated synthetic UAV telemetry
    - "flight_logs": real XAI logs from data/logs/xai_*.json
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import load_config
from .engine import XAIEngine
from .factory import create_default_engine
from ..utils import plot_tabular_attribution


# ============================================================
# JSON helper
# ============================================================
def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable."
    )


# ============================================================
# Synthetic data
# ============================================================
def _generate_synthetic_data(
    n_samples: int,
    seed: int,
    n_features: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic tabular data mimicking UAV telemetry.
    """

    rng = np.random.default_rng(seed)
    data, labels = [], []

    for _ in range(n_samples):
        altitude = rng.uniform(2, 30)
        speed = rng.uniform(1, 15)
        obstacle_dist = rng.uniform(2, 100)
        battery = rng.uniform(20, 100)
        density = rng.uniform(0, 1)
        yaw = rng.uniform(-180, 180)
        roll = rng.uniform(-30, 30)
        pitch = rng.uniform(-30, 30)
        dist_to_target = rng.uniform(10, 500)
        angle_to_target = rng.uniform(-180, 180)

        label = 0 if (obstacle_dist < 10 or battery < 15) else 1

        data.append([
            altitude, speed, obstacle_dist, battery, density,
            yaw, roll, pitch, dist_to_target, angle_to_target,
        ])
        labels.append(label)

    return np.array(data), np.array(labels)


# ============================================================
# Real flight logs
# ============================================================
def _load_flight_logs(
    log_dir: str | Path,
    feature_names: list[str],
    max_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load raw_state and action from xai_*.json logs.
    """

    log_dir = Path(log_dir)

    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    json_files = sorted(
        log_dir.glob("xai_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Skip tiny placeholder files
    json_files = [p for p in json_files if p.stat().st_size > 100]

    if max_files is not None:
        json_files = json_files[:max_files]

    if not json_files:
        raise FileNotFoundError(f"No usable xai_*.json files found in {log_dir}")

    print(f"      Found {len(json_files)} log files")

    X_list, y_list = [], []
    skipped_no_raw_state = 0
    skipped_other = 0

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"      ⚠️ Could not parse {json_file.name}: {e}")
            skipped_other += 1
            continue

        if isinstance(payload, dict):
            entries = payload.get("explanations") or payload.get("logs") or []
            if not isinstance(entries, list):
                entries = [payload]
        elif isinstance(payload, list):
            entries = payload
        else:
            skipped_other += 1
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                skipped_other += 1
                continue

            raw_state = entry.get("raw_state")
            if raw_state is None:
                skipped_no_raw_state += 1
                continue

            action = entry.get("action", None)
            if action is None:
                skipped_other += 1
                continue

            if isinstance(raw_state, dict):
                vector = [float(raw_state.get(name, 0.0)) for name in feature_names]
            elif isinstance(raw_state, list):
                if len(raw_state) != len(feature_names):
                    skipped_other += 1
                    continue
                vector = [float(v) for v in raw_state]
            else:
                skipped_other += 1
                continue

            X_list.append(vector)
            y_list.append(int(action))

    if len(X_list) == 0:
        raise ValueError(
            f"No entries with raw_state found in {log_dir}. "
            f"Run a new flight with the patched xai_bridge.py."
        )

    # ============================================================
    # АУГМЕНТАЦИЯ: если только один класс — добавляем синтетические
    # «плохие» состояния (класс 0)
    # ============================================================
    y_array = np.array(y_list, dtype=np.int64)

    if len(np.unique(y_array)) == 1:
        print(f"      ⚠️ Only one class ({y_array[0]}) in flight logs. "
              f"Augmenting with synthetic class 0 samples.")

        n_augment = max(20, len(X_list))
        rng = np.random.default_rng(42)

        for _ in range(n_augment):
            # «Плохое» состояние: низкая высота, близко препятствие, мало батареи
            bad_state = [
                rng.uniform(1.5, 3.0),      # altitude
                rng.uniform(0.5, 3.0),      # speed
                rng.uniform(2.0, 10.0),     # obstacle_dist
                rng.uniform(10.0, 25.0),    # battery
                rng.uniform(0.6, 1.0),      # obstacle_density
                rng.uniform(-180, 180),     # yaw
                rng.uniform(-30, 30),       # roll
                rng.uniform(-30, 30),       # pitch
                rng.uniform(10, 100),       # distance_to_target
                rng.uniform(-180, 180),     # angle_to_target
            ]
            X_list.append(bad_state)
            y_list.append(0)

        print(f"      Added {n_augment} synthetic class 0 samples. "
              f"Total: {len(X_list)}")

    print(f"      Loaded {len(X_list)} samples "
          f"({skipped_no_raw_state} without raw_state, "
          f"{skipped_other} other)")

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int64),
    )


# ============================================================
# Model training
# ============================================================
def _train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 50,
    random_state: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def _train_simple_nn(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    hidden_dim: int = 32,
    epochs: int = 100,
    lr: float = 0.005,
):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class SimpleNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)

    model = SimpleNN(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ============================================================
# Runner
# ============================================================
class TabularExperimentRunner:
    """
    Run a complete tabular XAI experiment from a YAML config.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)

        self.output_dir: Path | None = None
        self.engine: XAIEngine | None = None

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------
    def _create_output_dir(self) -> Path:
        experiment_name = self.config["experiment"]["name"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(self.config["output"]["root"])

        output_dir = root / f"{experiment_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)

        self.output_dir = output_dir
        return output_dir

    def _save_config(self, output_dir: Path) -> None:
        destination = output_dir / "config.yaml"
        shutil.copy2(self.config_path, destination)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    def _load_data(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        data_cfg = self.config["data"]
        source = data_cfg.get("source", "synthetic")

        feature_names = data_cfg.get("feature_names")
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(10)]
        feature_names = list(feature_names)

        if source == "synthetic":
            n_samples = int(data_cfg.get("n_samples", 500))
            seed = int(data_cfg.get("seed", 42))
            X, y = _generate_synthetic_data(n_samples, seed)

        elif source == "flight_logs":
            log_dir = data_cfg.get("log_dir", "data/logs")
            max_files = data_cfg.get("max_files", None)
            if max_files is not None:
                max_files = int(max_files)
            X, y = _load_flight_logs(
                log_dir=log_dir,
                feature_names=feature_names,
                max_files=max_files,
            )

        else:
            raise ValueError(
                f"Unsupported data.source: '{source}'. "
                f"Supported: 'synthetic', 'flight_logs'."
            )

        return X, y, feature_names

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    def _load_model(self, X: np.ndarray, y: np.ndarray):
        model_cfg = self.config["model"]
        name = model_cfg["name"].lower()

        if name == "random_forest":
            model = _train_random_forest(
                X, y,
                n_estimators=int(model_cfg.get("n_estimators", 50)),
                random_state=int(model_cfg.get("random_state", 42)),
            )
        elif name == "simple_nn":
            model = _train_simple_nn(
                X, y,
                input_dim=X.shape[1],
                hidden_dim=int(model_cfg.get("hidden_dim", 32)),
                epochs=int(model_cfg.get("epochs", 100)),
                lr=float(model_cfg.get("lr", 0.005)),
            )
        else:
            raise ValueError(
                f"Unsupported model.name: '{name}'. "
                f"Supported: 'random_forest', 'simple_nn'."
            )

        return model

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------
    def _resolve_target(self, model, test_vector: np.ndarray) -> int:
        target_cfg = self.config["xai"].get("target", "predicted")

        if target_cfg == "predicted":
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([test_vector])[0]
                return int(np.argmax(proba))
            import torch
            with torch.no_grad():
                out = model(torch.FloatTensor(test_vector).reshape(1, -1))
                return int(out.argmax(dim=1).item())
        else:
            return int(target_cfg)

    # --------------------------------------------------------
    # Main entry point
    # --------------------------------------------------------
    def run(self) -> dict[str, Any]:
        output_dir = self._create_output_dir()
        self._save_config(output_dir)

        print("=" * 60)
        print("UAV-XAI TABULAR EXPERIMENT")
        print("=" * 60)
        print(f"Config: {self.config_path}")
        print(f"Output: {output_dir}")

        # ----- Data -----
        print("\n[1/5] Loading data...")
        X, y, feature_names = self._load_data()
        print(f"      Shape: {X.shape}, features: {len(feature_names)}")
        print(f"      Class balance: {np.bincount(y).tolist()}")

        if len(X) < 5:
            raise ValueError(f"Too few samples: {len(X)}.")

        # ----- Model -----
        print("\n[2/5] Training model...")
        model = self._load_model(X, y)
        model_name = self.config["model"]["name"]
        print(f"      Model: {model_name}")

        # ----- Test instance -----
        test_idx = int(self.config["xai"].get("test_index", 0))
        if test_idx >= len(X):
            test_idx = 0
        test_vector = X[test_idx]
        print(f"      Test index: {test_idx}")

        # ----- Target -----
        target = self._resolve_target(model, test_vector)
        print(f"      Target class: {target}")

        # ----- Engine -----
        self.engine = create_default_engine()
        method = self.config["xai"]["method"]
        print(f"      Method: {method}")

        # ----- Explain -----
        print("\n[3/5] Generating explanation...")
        import torch as _torch

        if method == "lrp":
            input_data = _torch.FloatTensor(test_vector)
        else:
            input_data = test_vector

        explanation = self.engine.explain(
            input_type="tabular",
            method=method,
            model=model,
            input_data=input_data,
            target=target,
            training_data=X,
            feature_names=feature_names,
            class_names=self.config["xai"].get("class_names"),
            num_samples=int(self.config["xai"].get("num_samples", 1000)),
            background_data=X[:100],
            nsamples=int(self.config["xai"].get("nsamples", 100)),
        )

        print(f"      Runtime: {explanation.runtime_ms:.2f} ms")
        print(f"      Attribution shape: {explanation.attribution.shape}")

        # ----- Visualization -----
        print("\n[4/5] Saving visualization...")
        chart_path = output_dir / "attribution.png"
        plot_tabular_attribution(
            attribution=explanation.attribution,
            feature_names=feature_names,
            output_path=chart_path,
            title=f"{method.upper()} - {self.config['experiment']['name']}",
            normalize=True,
        )
        print(f"      Saved: {chart_path}")

        # ----- Result JSON -----
        print("\n[5/5] Saving result...")
        result: dict[str, Any] = {
            "experiment": self.config["experiment"]["name"],
            "timestamp": datetime.now().isoformat(),
            "data": {
                "source": self.config["data"].get("source", "synthetic"),
                "shape": list(X.shape),
                "feature_names": feature_names,
                "test_index": test_idx,
                "test_vector": test_vector.tolist(),
                "target": target,
            },
            "model": {
                "name": model_name,
            },
            "explanation": {
                "method": explanation.method,
                "target": explanation.target,
                "confidence": explanation.confidence,
                "runtime_ms": explanation.runtime_ms,
                "attribution": explanation.attribution.tolist(),
                "input_type": explanation.input_type,
                "feature_names": explanation.feature_names,
                "metadata": {
                    k: v
                    for k, v in explanation.metadata.items()
                    if k not in ("values",)
                },
            },
            "artifacts": {
                "attribution_chart": chart_path.name,
            },
        }

        result_path = output_dir / "result.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=_json_default)

        print(f"      Saved: {result_path}")

        if self.engine is not None:
            self.engine.close()

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)

        return result