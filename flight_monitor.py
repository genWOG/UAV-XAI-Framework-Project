"""
Главный модуль мониторинга полёта с XAI интеграцией

Версия 19.1
- async_move_to защищён lock'ом: одна команда движения в момент времени
- angle_to_target нормализуется в [-180, 180]
- жёсткий потолок высоты (MAX_ALTITUDE)
- обучение XAI-моделей ДО подключения к PX4
- CUDA отключена (GTX 1050 Ti не поддерживается текущим torch)
- реальный выход модели (predict_proba) для каждого отсчёта
- запись логов для post-hoc XAI-модуля (modules/incident_logger.py)
"""

import sys
import os

# Отключаем CUDA ДО любого импорта torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import time
import math
import random
import threading
import numpy as np
from datetime import datetime

if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'xai', 'modules'))

from modules.log_manager import LogManager
from modules.px4_adapter import PX4Adapter
from modules.incident_logger import IncidentLogger
# from modules.camera_reader import CameraReader  # Отключено — камера отдельно


# ========================================
# Неблокирующий move_to (защищённый lock'ом)
# ========================================
_move_lock = threading.Lock()


def async_move_to(adapter, x, y, z, speed=3.0):
    """Отправляет команду движения в фоне.

    Гарантирует, что одновременно работает только один move_to.
    Если во время движения приходит новая команда — она игнорируется
    (не сбиваем PX4 противоречивыми set_position_ned).
    """
    def _worker():
        # blocking=False: если кто-то уже двигается — пропускаем
        if not _move_lock.acquire(blocking=False):
            return
        try:
            adapter.move_to(x, y, z, speed)
        except Exception as e:
            # логирование ошибок делает сам адаптер
            pass
        finally:
            _move_lock.release()

    threading.Thread(target=_worker, daemon=True).start()


try:
    from xai.modules.xai_bridge import XAIBridge
    XAI_AVAILABLE = True
    print("✅ XAI Bridge загружен")
except ImportError as e:
    XAI_AVAILABLE = False
    print(f"⚠️ XAI Bridge не найден: {e}")


# ========================================
# XAI данные и модели
# ========================================

def generate_training_data(n_samples=500):
    np.random.seed(42)
    data, labels = [], []
    for _ in range(n_samples):
        altitude = np.random.uniform(2, 30)
        speed = np.random.uniform(1, 15)
        obstacle_dist = np.random.uniform(2, 100)
        battery = np.random.uniform(20, 100)
        density = np.random.uniform(0, 1)
        yaw = np.random.uniform(-180, 180)
        roll = np.random.uniform(-30, 30)
        pitch = np.random.uniform(-30, 30)
        dist_to_target = np.random.uniform(10, 500)
        angle_to_target = np.random.uniform(-180, 180)
        label = 0 if (obstacle_dist < 10 or battery < 15) else 1
        data.append([altitude, speed, obstacle_dist, battery, density,
                     yaw, roll, pitch, dist_to_target, angle_to_target])
        labels.append(label)
    return np.array(data), np.array(labels)


def train_xai_model(X, y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model


def train_torch_model(X, y):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class SimpleNN(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
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

    model = SimpleNN(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    model.train()
    for _ in range(200):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def pick_random_target(cx, cy, r_min=15.0, r_max=40.0, avoid_angle=None):
    if avoid_angle is not None:
        angle = avoid_angle + math.pi + random.uniform(-0.8, 0.8)
    else:
        angle = random.uniform(0, 2 * math.pi)
    radius = random.uniform(r_min, r_max)
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)


# ========================================
# Параметры
# ========================================

TOTAL_FLIGHT_SECONDS = 180
LOOP_SLEEP_SECONDS = 0.2   # цикл ~5 Гц
XAI_INTERVAL_SECONDS = 5
PLOT_INTERVAL_SECONDS = 30
FRAME_SAVE_INTERVAL = 10

TARGET_REACHED_RADIUS = 3.0
HOVER_AFTER_REACHED = 3
STUCK_THRESHOLD_SECONDS = 8

ALTITUDE_DROP_THRESHOLD = 1.5
SPEED_DROP_THRESHOLD = 0.3
FLIGHT_ALTITUDE = -4.5
MAX_ALTITUDE = 8.0
MIN_ALTITUDE = 3.0
FLIGHT_SPEED = 2.5


# ========================================
# Инициализация
# ========================================

print("\n" + "=" * 60)
print(f"🚁 ИССЛЕДОВАТЕЛЬ v19.1 ({TOTAL_FLIGHT_SECONDS} сек) + КАМЕРА")
print("=" * 60 + "\n")

log = LogManager(log_dir="data/logs")

# --- Тяжёлая часть (torch, обучение) ДО подключения к PX4 ---
training_data, training_labels = generate_training_data(500)
feature_names = [
    'altitude', 'speed', 'obstacle_dist', 'battery',
    'obstacle_density', 'yaw', 'roll', 'pitch',
    'distance_to_target', 'angle_to_target'
]

print("🎓 Обучение XAI-моделей (до подключения к PX4)...")
xai_model = train_xai_model(training_data, training_labels)
torch_model = train_torch_model(training_data, training_labels)
print("✅ Модели обучены")

xai = XAIBridge(log, training_data=training_data, feature_names=feature_names)
xai.set_model(xai_model)
xai.set_torch_model(torch_model)
xai_enabled = True

# --- Подключение к PX4 ---
adapter = PX4Adapter(log)
if not adapter.connect():
    log.error("❌ Не удалось подключиться к PX4")
    sys.exit(1)

# --- Логгер инцидентов ---
incident = IncidentLogger(
    root="data/incidents",
    meta={
        "script": "flight_monitor.py v19.1",
        "world": "uav_test_world",
        "flight_seconds": TOTAL_FLIGHT_SECONDS,
        "loop_sleep_seconds": LOOP_SLEEP_SECONDS,
        "feature_names": feature_names,
        "decision_model": "RandomForest(50), обучен на синтетике (временная заглушка)",
        "action_names": {"0": "Остановка", "1": "Продолжить"},
    },
)
log.system(f"📁 Логи инцидентов: {incident.dir}")

# ========================================
# КАМЕРА
# ========================================

camera = None
camera_enabled = False
log.system("📷 Камера работает в отдельном процессе (camera_recorder.py)")

log.system("Запуск исследователя...")
log.system(f"Время старта: {datetime.now().strftime('%H:%M:%S')}")


# ========================================
# Счётчики
# ========================================

session_start = time.time()
total_xai_explanations = []
total_xai_method_stats = {'lime': 0, 'shap': 0, 'lrp': 0, 'fallback': 0}
total_targets_reached = 0
total_stuck_events = 0
total_altitude_events = 0
total_speed_events = 0
total_ceiling_events = 0
total_frames_saved = 0


# ========================================
# Взлёт
# ========================================

log.ok("🚀 Разрешение на взлёт")
if not adapter.takeoff():
    log.error("❌ Не удалось взлететь")
    incident.event("takeoff_failed")
    incident.close(summary={"result": "takeoff_failed"})
    if camera is not None:
        camera.stop()
    sys.exit(1)
incident.event("takeoff")
time.sleep(3)

state = adapter.get_state()
pos = state['position']
target_x, target_y = pick_random_target(pos['x'], pos['y'])

log.drone(f"🎯 Первая цель: ({target_x:.1f}, {target_y:.1f})")
async_move_to(adapter, target_x, target_y, FLIGHT_ALTITUDE, FLIGHT_SPEED)

# ========================================
# Счётчики РЕАЛЬНОГО времени
# ========================================

last_xai_real_time = 0.0
last_plot_real_time = 0.0
last_status_time = time.time()
last_frame_time = time.time()

last_pos = (pos['x'], pos['y'])
last_move_time = time.time()
hover_start = None
last_altitude_event_time = 0
last_speed_event_time = 0


# ========================================
# Основной цикл
# ========================================

try:
    while (time.time() - session_start) < TOTAL_FLIGHT_SECONDS:
        time.sleep(LOOP_SLEEP_SECONDS)

        state = adapter.get_state()
        if state is None:
            continue

        pos = state['position']
        vel = state['velocity']
        now = time.time()
        elapsed = now - session_start
        cur_altitude = abs(pos['z'])
        cur_speed = math.sqrt(vel['x']**2 + vel['y']**2 + vel['z']**2)

        # ========================================
        # Жёсткий потолок высоты
        # ========================================
        if cur_altitude > MAX_ALTITUDE + 0.5:
            total_ceiling_events += 1
            log.warn(f"⚠️ Потолок: {cur_altitude:.1f} м > {MAX_ALTITUDE} м — принудительно вниз")
            incident.event("altitude_ceiling",
                           altitude=cur_altitude,
                           max_altitude=MAX_ALTITUDE,
                           position=[pos['x'], pos['y'], pos['z']])
            async_move_to(adapter, pos['x'], pos['y'], -MAX_ALTITUDE, FLIGHT_SPEED)
            last_move_time = now
            hover_start = None
            continue

        # ========================================
        # Признаки цели + РЕАЛЬНЫЙ выход модели (каждый проход)
        # ========================================
        model_action, model_conf, model_proba, model_features = None, None, None, None
        try:
            dx_t = target_x - pos['x']
            dy_t = target_y - pos['y']
            dist_t = math.sqrt(dx_t * dx_t + dy_t * dy_t)
            yaw = state['orientation'].get('yaw', 0)

            # Нормализация угла в [-180, 180] (ФИКС)
            angle_t = math.degrees(math.atan2(dy_t, dx_t)) - yaw
            angle_t = (angle_t + 180.0) % 360.0 - 180.0

            state['target_x'] = target_x
            state['target_y'] = target_y
            state['distance_to_target'] = dist_t
            state['angle_to_target'] = angle_t

            model_features = xai._extract_features(state)
            model_proba = xai_model.predict_proba(model_features.reshape(1, -1))[0]
            model_action = int(np.argmax(model_proba))
            model_conf = float(model_proba[model_action])

            incident.model_output(
                model="random_forest_synthetic",
                feature_names=feature_names,
                features=model_features,
                proba=model_proba,
                action=model_action,
                confidence=model_conf,
            )
        except Exception as e:
            log.error(f"❌ Ошибка модели решений: {e}")

        # Телеметрия с таймстемпом
        incident.telemetry(state, extra={
            "target": [target_x, target_y],
            "distance_to_target": state.get('distance_to_target'),
        })

        # ========================================
        # XAI (раз в XAI_INTERVAL_SECONDS реального времени)
        # ========================================
        if xai_enabled and model_action is not None and \
           (now - last_xai_real_time) >= XAI_INTERVAL_SECONDS:
            last_xai_real_time = now
            try:
                if camera_enabled:
                    frame = camera.get_latest_frame()
                    if frame is not None:
                        state['frame_mean'] = float(frame.mean())
                        state['frame_std'] = float(frame.std())

                explanation = xai.explain_decision(
                    state=state, action=model_action,
                    confidence=model_conf, method="auto"
                )

                if explanation:
                    total_xai_explanations.append(explanation)
                    m = explanation.get('method', 'unknown')
                    total_xai_method_stats[m] = total_xai_method_stats.get(m, 0) + 1
                    log.info(f"📊 XAI[{m}]: "
                             f"conf={explanation.get('confidence', 0):.2f}, "
                             f"time={explanation.get('runtime_ms', 0):.1f}ms")
            except Exception as e:
                log.error(f"❌ XAI ошибка: {e}")

        # ========================================
        # Проверка движения (застревание)
        # ========================================
        dx = pos['x'] - last_pos[0]
        dy = pos['y'] - last_pos[1]
        moved = math.sqrt(dx * dx + dy * dy)
        if moved > 0.3:
            last_move_time = now
            stuck_seconds = 0
        else:
            stuck_seconds = now - last_move_time
        last_pos = (pos['x'], pos['y'])

        dist_to_target = math.sqrt(
            (pos['x'] - target_x) ** 2 + (pos['y'] - target_y) ** 2
        )

        # Долетели?
        if dist_to_target < TARGET_REACHED_RADIUS and hover_start is None:
            hover_start = now
            total_targets_reached += 1
            log.ok(f"🎯 Цель #{total_targets_reached}. Висим {HOVER_AFTER_REACHED} сек")
            incident.event("target_reached", n=total_targets_reached,
                           position=[pos['x'], pos['y'], pos['z']])

        if hover_start is not None and (now - hover_start) >= HOVER_AFTER_REACHED:
            target_x, target_y = pick_random_target(pos['x'], pos['y'])
            log.drone(f"🎯 Новая цель: ({target_x:.1f}, {target_y:.1f})")
            incident.event("new_target", target=[target_x, target_y])
            async_move_to(adapter, target_x, target_y, FLIGHT_ALTITUDE, FLIGHT_SPEED)
            hover_start = None
            last_move_time = now

        # Просадка высоты
        target_altitude = abs(FLIGHT_ALTITUDE)
        altitude_drop = target_altitude - cur_altitude
        if altitude_drop > ALTITUDE_DROP_THRESHOLD and \
           (now - last_altitude_event_time) > 5:
            last_altitude_event_time = now
            total_altitude_events += 1
            log.warn(f"⚠️ Просадка высоты: {cur_altitude:.1f} м "
                     f"— предполагаемое препятствие снизу #{total_altitude_events}")
            incident.event("altitude_drop", altitude=cur_altitude,
                           expected=target_altitude, position=[pos['x'], pos['y'], pos['z']],
                           model_action=model_action, model_confidence=model_conf)
            target_x, target_y = pick_random_target(pos['x'], pos['y'])
            async_move_to(adapter, target_x, target_y, FLIGHT_ALTITUDE - 2.0, FLIGHT_SPEED)
            last_move_time = now
            hover_start = None
            continue

        # Потеря скорости
        if cur_speed < SPEED_DROP_THRESHOLD and dist_to_target > 5.0 and \
           hover_start is None and (now - last_speed_event_time) > 5:
            last_speed_event_time = now
            total_speed_events += 1
            dx_t = target_x - pos['x']
            dy_t = target_y - pos['y']
            stuck_angle = math.atan2(dy_t, dx_t)
            incident.event("speed_loss", speed=cur_speed, distance_to_target=dist_to_target,
                           position=[pos['x'], pos['y'], pos['z']],
                           model_action=model_action, model_confidence=model_conf)
            log.warn(f"⚠️ Скорость {cur_speed:.2f} м/с при расстоянии "
                     f"{dist_to_target:.1f} м — предполагаемое препятствие впереди "
                     f"#{total_speed_events}")
            target_x, target_y = pick_random_target(
                pos['x'], pos['y'], avoid_angle=stuck_angle
            )
            async_move_to(adapter, target_x, target_y, FLIGHT_ALTITUDE, FLIGHT_SPEED)
            last_move_time = now
            continue

        # Застревание
        if stuck_seconds >= STUCK_THRESHOLD_SECONDS and hover_start is None:
            total_stuck_events += 1
            dx_t = target_x - pos['x']
            dy_t = target_y - pos['y']
            stuck_angle = math.atan2(dy_t, dx_t)
            log.warn(f"⚠️ Застрял {stuck_seconds:.0f} сек #{total_stuck_events}")
            incident.event("stuck", seconds=stuck_seconds,
                           position=[pos['x'], pos['y'], pos['z']],
                           model_action=model_action, model_confidence=model_conf)
            target_x, target_y = pick_random_target(
                pos['x'], pos['y'], avoid_angle=stuck_angle
            )
            async_move_to(adapter, target_x, target_y, FLIGHT_ALTITUDE, FLIGHT_SPEED)
            last_move_time = now

        # === СОХРАНЕНИЕ КАДРА КАМЕРЫ (отключено) ===
        # if camera_enabled and (now - last_frame_time) >= FRAME_SAVE_INTERVAL:
        #     ...

        # === График (раз в PLOT_INTERVAL_SECONDS реального времени) ===
        if total_xai_explanations and (now - last_plot_real_time) >= PLOT_INTERVAL_SECONDS:
            last_plot_real_time = now
            try:
                exp = total_xai_explanations[-1]
                attr = np.array(exp.get('attribution', []))
                fn = exp.get('feature_names', [])
                m = exp.get('method', 'unknown')
                if len(fn) == len(attr) and len(attr) > 0:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    colors = ['green' if v > 0 else 'red' for v in attr]
                    plt.barh(fn, attr, color=colors, alpha=0.7)
                    plt.title(f'XAI ({m.upper()}, t={elapsed:.0f}s)')
                    plt.xlabel('Вклад')
                    plt.axvline(0, color='black', linewidth=0.5)
                    plt.tight_layout()
                    plot_dir = "results/xai_plots"
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_path = os.path.join(plot_dir, f"xai_{m}_t{int(elapsed):04d}.png")
                    plt.savefig(plot_path, dpi=120)
                    plt.close()
                    log.info(f"📊 График: {plot_path}")
            except Exception as e:
                log.warn(f"⚠️ Ошибка графика: {e}")

        # === Статус ===
        if (now - last_status_time) >= 5:
            last_status_time = now
            frame_info = ""
            if camera_enabled:
                frame_info = f" | 📷 {camera.frames_count}"
            log.drone(f"⏱️ {elapsed:.0f}с | "
                      f"📍 ({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f}) | "
                      f"🎯 {dist_to_target:.1f}м | "
                      f"⚡ {cur_speed:.2f}м/с | "
                      f"⏳ {stuck_seconds:.0f}с{frame_info}")

except KeyboardInterrupt:
    log.warn("⏹️ Прервано пользователем")
except Exception as e:
    log.error(f"❌ Критическая ошибка: {e}")
    import traceback
    traceback.print_exc()


# ========================================
# Посадка
# ========================================

log.drone("🛬 Посадка...")
incident.event("landing")
adapter.land()
time.sleep(5)

if camera is not None:
    camera.stop()


# ========================================
# Отчёт
# ========================================

session_time = time.time() - session_start
print("\n" + "=" * 60)
print("📊 ОТЧЁТ")
print("=" * 60)

log.info(f"⏱️ Время: {session_time:.0f} сек ({session_time/60:.1f} мин)")
log.info(f"🎯 Целей достигнуто: {total_targets_reached}")
log.info(f"⏳ Застреваний: {total_stuck_events}")
log.info(f"📉 Просадок высоты: {total_altitude_events}")
log.info(f"🐢 Потерь скорости: {total_speed_events}")
log.info(f"📈 Срабатываний потолка: {total_ceiling_events}")
log.info(f"📷 Кадров сохранено: {total_frames_saved}")
log.info(f"📊 XAI: {len(total_xai_explanations)}")

for m, count in total_xai_method_stats.items():
    if count > 0:
        pct = count / max(len(total_xai_explanations), 1) * 100
        log.info(f"   {m.upper()}: {count} ({pct:.1f}%)")

if total_xai_explanations:
    p = f"data/logs/xai_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        xai.save_explanations(p)
        log.info(f"💾 XAI: {p}")
    except Exception as e:
        log.warn(f"⚠️ {e}")

incident.close(summary={
    "targets_reached": total_targets_reached,
    "stuck_events": total_stuck_events,
    "altitude_drop_events": total_altitude_events,
    "speed_loss_events": total_speed_events,
    "ceiling_events": total_ceiling_events,
    "xai_explanations": len(total_xai_explanations),
    "xai_method_stats": total_xai_method_stats,
})
log.info(f"📁 Инцидентные логи: {incident.dir}")

log.save()
adapter.disconnect()
print("\n✅ Готово!")