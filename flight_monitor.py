"""
Главный модуль мониторинга полёта с XAI интеграцией.
Версия 22.0 — LiDAR + безопасный обход + отступление из "мешка".
"""

import sys
import os
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
from modules.lidar_reader import LidarReader

# ========================================
# Неблокирующий move_to
# ========================================
_move_lock = threading.Lock()

def async_move_to(adapter, x, y, z, speed=3.0):
    def _worker():
        if not _move_lock.acquire(blocking=False):
            return
        try:
            adapter.move_to(x, y, z, speed)
        except Exception:
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

def pick_random_target(cx, cy, r_min=10.0, r_max=25.0, avoid_angle=None):
    if avoid_angle is not None:
        angle = avoid_angle + math.pi + random.uniform(-0.8, 0.8)
    else:
        angle = random.uniform(0, 2 * math.pi)
    radius = random.uniform(r_min, r_max)
    return cx + radius * math.cos(angle), cy + radius * math.sin(angle)

# ========================================
# Параметры
# ========================================
TOTAL_FLIGHT_SECONDS = 300
LOOP_SLEEP_SECONDS = 0.2
XAI_INTERVAL_SECONDS = 5
PLOT_INTERVAL_SECONDS = 30
TARGET_REACHED_RADIUS = 3.0
HOVER_AFTER_REACHED = 3
STUCK_THRESHOLD_SECONDS = 8
ALTITUDE_DROP_THRESHOLD = 1.5
SPEED_DROP_THRESHOLD = 0.3
FLIGHT_ALTITUDE = -4.5
MAX_ALTITUDE = 8.0
MIN_ALTITUDE = 2.0
FLIGHT_SPEED = 2.5

# Пороги LiDAR
LIDAR_OBSTACLE_WARN = 5.0      # ближе 5 м — препятствие
LIDAR_OBSTACLE_CRIT = 2.5      # ближе 2.5 м — критично
LIDAR_AVOID_DIST = 8.0         # начинаем обход за 8 м
LIDAR_TRAP_DIST = 3.0          # "мешок": все три < 3 м
LIDAR_MIN_AVOID_INTERVAL = 3.0 # не чаще обхода раз в 3 сек

# ========================================
# Инициализация
# ========================================
print("\n" + "=" * 60)
print(f"🚁 ИССЛЕДОВАТЕЛЬ v22.0 ({TOTAL_FLIGHT_SECONDS} сек) + LiDAR + Safe Avoid")
print("=" * 60 + "\n")

log = LogManager(log_dir="data/logs")

training_data, training_labels = generate_training_data(500)
feature_names = [
    'altitude', 'speed', 'obstacle_dist', 'battery',
    'obstacle_density', 'yaw', 'roll', 'pitch',
    'distance_to_target', 'angle_to_target'
]

print("🎓 Обучение XAI-моделей...")
xai_model = train_xai_model(training_data, training_labels)
torch_model = train_torch_model(training_data, training_labels)
print("✅ Модели обучены")

xai = XAIBridge(log, training_data=training_data, feature_names=feature_names)
xai.set_model(xai_model)
xai.set_torch_model(torch_model)
xai_enabled = True

adapter = PX4Adapter(log)
if not adapter.connect():
    log.error("❌ Не удалось подключиться к PX4")
    sys.exit(1)

incident = IncidentLogger(
    root="data/incidents",
    meta={
        "script": "flight_monitor.py v22.0",
        "world": "uav_test_world",
        "flight_seconds": TOTAL_FLIGHT_SECONDS,
        "loop_sleep_seconds": LOOP_SLEEP_SECONDS,
        "feature_names": feature_names,
        "decision_model": "RandomForest(50), обучен на синтетике",
        "action_names": {"0": "Остановка", "1": "Продолжить"},
    },
)
log.system(f"📁 Логи инцидентов: {incident.dir}")

# LiDAR
lidar = LidarReader(log)
lidar_enabled = lidar.start()
if lidar_enabled:
    adapter.set_lidar(lidar)
    log.system("📡 LiDAR подключён к адаптеру")
else:
    log.warn("⚠️ LiDAR не подключён — дрон летает вслепую")

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
total_obstacle_events = 0
total_trap_events = 0

# ========================================
# Взлёт
# ========================================
log.ok("🚀 Разрешение на взлёт")
if not adapter.takeoff():
    log.error("❌ Не удалось взлететь")
    incident.event("takeoff_failed")
    incident.close(summary={"result": "takeoff_failed"})
    if lidar is not None:
        lidar.stop()
    sys.exit(1)
incident.event("takeoff")
time.sleep(3)

state = adapter.get_state()
pos = state['position']
target_x, target_y = pick_random_target(pos['x'], pos['y'])
log.drone(f"🎯 Первая цель: ({target_x:.1f}, {target_y:.1f})")
async_move_to(adapter, target_x, target_y, FLIGHT_ALTITUDE, FLIGHT_SPEED)

last_xai_real_time = 0.0
last_plot_real_time = 0.0
last_status_time = time.time()
last_pos = (pos['x'], pos['y'])
last_move_time = time.time()
hover_start = None
last_altitude_event_time = 0
last_speed_event_time = 0
last_obstacle_event_time = 0

# Запоминаем "безопасную" позицию (где все LiDAR > 8 м)
last_safe_pos = (pos['x'], pos['y'])

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
        orient = state['orientation']
        sensors = state['sensors']
        now = time.time()
        elapsed = now - session_start
        cur_altitude = abs(pos['z'])
        cur_speed = math.sqrt(vel['x']**2 + vel['y']**2 + vel['z']**2)

        # --- Проверка ориентации ---
        roll = orient.get('roll', 0)
        pitch = orient.get('pitch', 0)
        if abs(roll) > 90 or abs(pitch) > 90:
            log.error(f"❌ Дрон перевернулся! roll={roll:.1f}, pitch={pitch:.1f}")
            incident.event("flip", roll=roll, pitch=pitch,
                           position=[pos['x'], pos['y'], pos['z']])
            break

        # --- Потолок ---
        if cur_altitude > MAX_ALTITUDE + 0.5:
            total_ceiling_events += 1
            log.warn(f"⚠️ Потолок: {cur_altitude:.1f} м > {MAX_ALTITUDE} м — вниз")
            incident.event("altitude_ceiling", altitude=cur_altitude,
                           max_altitude=MAX_ALTITUDE,
                           position=[pos['x'], pos['y'], pos['z']])
            async_move_to(adapter, pos['x'], pos['y'], -MAX_ALTITUDE, FLIGHT_SPEED)
            last_move_time = now
            hover_start = None
            continue

        # --- LiDAR: проверка препятствий ---
        front = sensors.get('lidar_front', 100.0)
        left = sensors.get('lidar_left', 100.0)
        right = sensors.get('lidar_right', 100.0)
        back = sensors.get('lidar_back', 100.0)
        min_dist = sensors.get('obstacle_distance', 100.0)
        density = sensors.get('obstacle_density', 0.0)

        # Запоминаем "безопасную" позицию, если вокруг свободно
        if front > 8.0 and left > 8.0 and right > 8.0 and back > 8.0:
            last_safe_pos = (pos['x'], pos['y'])

        # Логируем LiDAR раз в 5 секунд
        if int(elapsed) % 5 == 0 and (now - last_status_time) >= 4.5:
            log.info(f"📡 LiDAR: вперёд={front:.1f}м, "
                     f"слева={left:.1f}м, справа={right:.1f}м, "
                     f"сзади={back:.1f}м, мин={min_dist:.1f}м, плотность={density:.2f}")

        # ============================================================
        # БЛОК 1: Проверка "мешка" (западня со всех сторон)
        # ============================================================
        in_trap = (front < LIDAR_TRAP_DIST and
                   left < LIDAR_TRAP_DIST and
                   right < LIDAR_TRAP_DIST)

        if in_trap and hover_start is None and \
           (now - last_obstacle_event_time) > LIDAR_MIN_AVOID_INTERVAL:
            total_trap_events += 1
            last_obstacle_event_time = now

            # Отступаем к последней безопасной позиции
            safe_x, safe_y = last_safe_pos
            log.warn(f"🚨 МЕШОК! вперёд={front:.1f}, слева={left:.1f}, "
                     f"справа={right:.1f} → ОТСТУП к ({safe_x:.1f}, {safe_y:.1f})")
            incident.event("trap_retreat",
                           front=front, left=left, right=right, back=back,
                           safe_pos=[safe_x, safe_y],
                           position=[pos['x'], pos['y'], pos['z']])

            target_x, target_y = safe_x, safe_y
            async_move_to(adapter, target_x, target_y, FLIGHT_ALTITUDE, FLIGHT_SPEED)
            last_move_time = now
            hover_start = None
            continue

        # ============================================================
        # БЛОК 2: Обычный обход препятствия (но не чаще раза в 3 сек)
        # ============================================================
        if front < LIDAR_AVOID_DIST and hover_start is None and \
           (now - last_obstacle_event_time) > LIDAR_MIN_AVOID_INTERVAL:
            total_obstacle_events += 1
            last_obstacle_event_time = now

            # Приоритет: если сзади больше места, чем сбоку — отступаем
            if back > max(left, right) and back > LIDAR_AVOID_DIST:
                side = "назад"
                angle_to_target = math.atan2(
                    pos['y'] - target_y, pos['x'] - target_x
                )
                new_angle = angle_to_target  # летим от цели
            elif left > right:
                side = "слева"
                new_angle = math.atan2(
                    pos['y'] - target_y, pos['x'] - target_x
                ) + math.radians(60)
            else:
                side = "справа"
                new_angle = math.atan2(
                    pos['y'] - target_y, pos['x'] - target_x
                ) - math.radians(60)

            radius = 15.0
            avoid_x = pos['x'] + radius * math.cos(new_angle)
            avoid_y = pos['y'] + radius * math.sin(new_angle)

            log.warn(f"⚠️ ПРЕПЯТСТВИЕ ВПЕРЕДИ! {front:.1f} м | "
                     f"Обход {side} → ({avoid_x:.1f}, {avoid_y:.1f})")
            incident.event("obstacle_avoid",
                           front=front, left=left, right=right,
                           back=back, side=side,
                           avoid_x=avoid_x, avoid_y=avoid_y,
                           position=[pos['x'], pos['y'], pos['z']])

            target_x, target_y = avoid_x, avoid_y
            async_move_to(adapter, target_x, target_y,
                          FLIGHT_ALTITUDE, FLIGHT_SPEED)
            last_move_time = now

        # --- Признаки цели + выход модели ---
        model_action, model_conf = None, None
        try:
            dx_t = target_x - pos['x']
            dy_t = target_y - pos['y']
            dist_t = math.sqrt(dx_t * dx_t + dy_t * dy_t)
            yaw = orient.get('yaw', 0)
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

        incident.telemetry(state, extra={
            "target": [target_x, target_y],
            "distance_to_target": state.get('distance_to_target'),
        })

        # --- XAI ---
        if xai_enabled and model_action is not None and \
           (now - last_xai_real_time) >= XAI_INTERVAL_SECONDS:
            last_xai_real_time = now
            try:
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

        # --- Проверка движения ---
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

        # Застревание
        if stuck_seconds >= STUCK_THRESHOLD_SECONDS and hover_start is None:
            total_stuck_events += 1
            dx_t = target_x - pos['x']
            dy_t = target_y - pos['y']
            stuck_angle = math.atan2(dy_t, dx_t)
            log.warn(f"⚠️ Застрял {stuck_seconds:.0f} сек #{total_stuck_events}")
            incident.event("stuck", seconds=stuck_seconds,
                           position=[pos['x'], pos['y'], pos['z']])
            target_x, target_y = pick_random_target(
                pos['x'], pos['y'], avoid_angle=stuck_angle
            )
            async_move_to(adapter, target_x, target_y, FLIGHT_ALTITUDE, FLIGHT_SPEED)
            last_move_time = now

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

if lidar is not None:
    lidar.stop()

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
log.info(f"🚧 Событий обхода препятствий: {total_obstacle_events}")
log.info(f"🚨 Событий 'мешок-отступление': {total_trap_events}")
log.info(f"📊 XAI: {len(total_xai_explanations)}")

for m, count in total_xai_method_stats.items():
    if count > 0:
        pct = count / max(len(total_xai_explanations), 1) * 100
        log.info(f"  {m.upper()}: {count} ({pct:.1f}%)")

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
    "obstacle_events": total_obstacle_events,
    "trap_events": total_trap_events,
    "xai_explanations": len(total_xai_explanations),
    "xai_method_stats": total_xai_method_stats,
})
log.info(f"📁 Инцидентные логи: {incident.dir}")
log.save()
adapter.disconnect()
print("\n✅ Готово!")