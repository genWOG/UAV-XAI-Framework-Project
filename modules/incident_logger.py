"""
IncidentLogger — сохранение всех логов полёта в один session.json.
"""

import json
import os
import time
from datetime import datetime

try:
    import numpy as np
except ImportError:
    np = None


def _jsonable(obj):
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


class IncidentLogger:
    def __init__(self, root="data/incidents", meta=None):
        self.t0 = time.time()
        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, self.session_id)
        os.makedirs(self.dir, exist_ok=True)

        # ВАЖНО: названия полей НЕ должны совпадать с именами методов!
        self.events_log = []
        self.telemetry_log = []
        self.model_outputs_log = []
        self._meta = _jsonable(meta or {})
        self._closed = False

    def telemetry(self, state, extra=None):
        now = time.time()
        rec = {
            "t": now,
            "elapsed": now - self.t0,
            "position": state.get("position"),
            "velocity": state.get("velocity"),
            "orientation": state.get("orientation"),
            "sensors": state.get("sensors"),
        }
        if extra:
            rec.update(extra)
        self.telemetry_log.append(_jsonable(rec))

    def model_output(self, model, feature_names, features, proba,
                     action, confidence, extra=None):
        rec = {
            "t": time.time(),
            "model": model,
            "feature_names": list(feature_names),
            "features": features,
            "proba": proba,
            "action": int(action),
            "confidence": float(confidence),
        }
        if extra:
            rec.update(extra)
        self.model_outputs_log.append(_jsonable(rec))

    def event(self, kind, **data):
        rec = {"t": time.time(), "kind": kind}
        rec.update(data)
        self.events_log.append(_jsonable(rec))

    def close(self, summary=None):
        if self._closed:
            return
        self._closed = True

        payload = {
            "meta": self._meta,
            "started_at": self.t0,
            "ended_at": time.time(),
            "duration_s": time.time() - self.t0,
            "events": self.events_log,
            "telemetry": self.telemetry_log,
            "model_outputs": self.model_outputs_log,
            "summary": _jsonable(summary or {}),
        }

        out_path = os.path.join(self.dir, "session.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[INCIDENT] Session saved: {out_path}")