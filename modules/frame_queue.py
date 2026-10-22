import os
import time
import json

REQUESTS_DIR = "data/incidents/requests"

def request_frame(event_kind: str, event_data: dict = None):
    """Вызывается из flight_monitor.py при событии."""
    os.makedirs(REQUESTS_DIR, exist_ok=True)
    filename = f"{time.time():.3f}_{event_kind}.json"
    filepath = os.path.join(REQUESTS_DIR, filename)
    try:
        with open(filepath, "w") as f:
            json.dump(event_data or {}, f)
    except Exception as e:
        print(f"⚠️ Не удалось создать запрос на кадр: {e}")

def get_pending_request():
    """Вызывается из camera_recorder.py."""
    if not os.path.exists(REQUESTS_DIR):
        return None
    files = sorted([f for f in os.listdir(REQUESTS_DIR) if f.endswith(".json")])
    if not files:
        return None
    
    filepath = os.path.join(REQUESTS_DIR, files[0])
    event_kind = files[0].split("_", 1)[1].replace(".json", "")
    
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
    except:
        data = {}
        
    return {"filename": filepath, "event_kind": event_kind, "data": data}

def remove_request(filepath: str):
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass