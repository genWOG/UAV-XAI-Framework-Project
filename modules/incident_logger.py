"""
IncidentLogger — запись логов полёта в формате для post-hoc XAI-модуля.

Структура сессии (одна папка на запуск):
    data/incidents/session_YYYYmmdd_HHMMSS/
        telemetry.jsonl      # одна строка = один отсчёт телеметрии (с таймстемпом)
        model_outputs.jsonl  # выходы модели: признаки, вероятности, действие
        events.jsonl         # события: цель достигнута, просадка, потеря скорости, ...
        manifest.json        # метаданные сессии + итоговая сводка

Все записи содержат поле "t" — время в секундах Unix (time.time()).
Кадры камеры сохраняются отдельным процессом (camera_recorder.py) в
results/frames/ с индексом frames_index.csv (колонки: t, filename);
связь с телеметрией — по полю "t".

JSONL выбран намеренно: файл дописывается построчно, поэтому полёт на часы
не держит всё в памяти и не теряет данные при аварийном завершении.
"""

import json
import os
import time
from datetime import datetime

try:
    import numpy as np
except ImportError:  # логгер не должен падать без numpy
    np = None


def _jsonable(obj):
    """Приводит numpy/прочие типы к JSON-совместимым."""
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
        # NaN/Inf не валидны в JSON
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


class IncidentLogger:
    FLUSH_EVERY = 20  # записей

    def __init__(self, root="data/incidents", meta=None):
        self.t0 = time.time()
        self.session_id = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, self.session_id)
        os.makedirs(self.dir, exist_ok=True)

        self._files = {
            "telemetry": open(os.path.join(self.dir, "telemetry.jsonl"), "a", encoding="utf-8"),
            "model_outputs": open(os.path.join(self.dir, "model_outputs.jsonl"), "a", encoding="utf-8"),
            "events": open(os.path.join(self.dir, "events.jsonl"), "a", encoding="utf-8"),
        }
        self._counts = {"telemetry": 0, "model_outputs": 0, "events": 0}
        self._event_kinds = {}
        self._meta = _jsonable(meta or {})
        self._closed = False
        self._write_manifest(final=False)

    # ------------------------------------------------------------------
    def _write(self, stream, record):
        if self._closed:
            return
        f = self._files[stream]
        f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
        self._counts[stream] += 1
        if self._counts[stream] % self.FLUSH_EVERY == 0:
            f.flush()

    # ------------------------------------------------------------------
    def telemetry(self, state, extra=None):
        """Один отсчёт телеметрии. state — словарь из PX4Adapter.get_state()."""
        now = time.time()
        rec = {
            "t": now,
            "elapsed": now - self.t0,
            "position": state.get("position"),
            "velocity": state.get("velocity"),
            "orientation": state.get("orientation"),
            # sensors пишем целиком: батарея, GPS, IMU, барометр, магнитометр —
            # всё, что отдаёт адаптер
            "sensors": state.get("sensors"),
        }
        if extra:
            rec.update(extra)
        self._write("telemetry", rec)

    def model_output(self, model, feature_names, features, proba, action,
                     confidence, extra=None):
        """Выход модели принятия решений (реальный, не константа)."""
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
        self._write("model_outputs", rec)

    def event(self, kind, **data):
        """Событие полёта (кандидат в «инцидент»)."""
        self._event_kinds[kind] = self._event_kinds.get(kind, 0) + 1
        rec = {"t": time.time(), "kind": kind}
        rec.update(data)
        self._write("events", rec)
        self._files["events"].flush()  # события редкие — сбрасываем сразу

    # ------------------------------------------------------------------
    def _write_manifest(self, final, summary=None):
        manifest = {
            "session_id": self.session_id,
            "started_at": self.t0,
            "meta": self._meta,
            "streams": {
                "telemetry": "telemetry.jsonl",
                "model_outputs": "model_outputs.jsonl",
                "events": "events.jsonl",
                "frames": "results/frames/frames_index.csv (колонки: t, filename)",
            },
            "counts": self._counts,
            "event_kinds": self._event_kinds,
            "finished": final,
        }
        if final:
            manifest["ended_at"] = time.time()
            manifest["duration_s"] = manifest["ended_at"] - self.t0
        if summary:
            manifest["summary"] = _jsonable(summary)
        with open(os.path.join(self.dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(_jsonable(manifest), f, ensure_ascii=False, indent=2)

    def close(self, summary=None):
        if self._closed:
            return
        for f in self._files.values():
            f.flush()
            f.close()
        self._closed = True
        self._write_manifest(final=True, summary=summary)
