"""
Главный модуль мониторинга полёта с XAI интеграцией

Версия 14.3 — ограничение высоты + GPS
"""

import sys
import os
import time
import json
import math
import random
import numpy as np
from datetime import datetime

# ========================================
# ФИКС КОДИРОВКИ ДЛЯ ПОДДЕРЖКИ ЭМОДЗИ В WINDOWS
# ========================================
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# Добавляем пути
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'xai', 'modules'))

from modules.log_manager import LogManager
from modules.obstacle_detector import ObstacleDetector
from modules.telemetry_parser import TelemetryParser
from modules.route_planner import RoutePlanner
from modules.airsim_adapter import AirSimAdapter
from modules.world_scanner import WorldScanner

# XAI Bridge
try:
    from xai.modules.xai_bridge import XAIBridge
    XAI_AVAILABLE = True
    print("✅ XAI Bridge загружен")
except ImportError as e:
    XAI_AVAILABLE = False
    print(f"⚠️ XAI Bridge не найден: {e}")

# ========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ========================================

def generate_training_data(n_samples=300):
    np.random.seed(42)
    data = []
    labels = []
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

        if obstacle_dist < 10 or battery < 15:
            label = 0
        else:
            label = 1
        data.append([altitude, speed, obstacle_dist, battery, density, yaw, roll, pitch, dist_to_target, angle_to_target])
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
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model

def estimate_obstacle_size(sensors):
    distance = sensors.get('obstacle_distance', 100.0)
    density = sensors.get('obstacle_density', 0.0)
    if distance < 5: return 50
    elif distance < 15: return 30
    elif distance < 30: return 20
    elif density > 0.8: return 50
    elif density > 0.5: return 25
    else: return 15

def classify_object(distance, width, speed):
    if width > 30:
        size = "ОГРОМНЫЙ"
        size_desc = "здание, большое сооружение"
    elif width > 15:
        size = "БОЛЬШОЙ"
        size_desc = "крупный объект"
    elif width > 8:
        size = "СРЕДНИЙ"
        size_desc = "средний объект"
    else:
        size = "МАЛЕНЬКИЙ"
        size_desc = "маленький объект"

    if speed < 0.5:
        motion = "СТОИТ"
        motion_desc = "неподвижный объект"
    else:
        motion = "ДВИЖЕТСЯ"
        motion_desc = f"движется со скоростью {speed:.1f} м/с"

    return {
        'size': size, 'size_desc': size_desc,
        'motion': motion, 'motion_desc': motion_desc,
        'full_desc': f"{size_desc} ({size}), {motion_desc}"
    }

def calculate_new_target(pos, obstacle_width, current_target_x, current_target_y, attempt=0):
    dx = current_target_x - pos['x']
    dy = current_target_y - pos['y']
    base_offset = max(obstacle_width / 2 + 40, 50)
    offset = base_offset + (attempt * 25)

    if attempt > 5:
        if dx > 0: new_x = current_target_x + offset + 150
        else: new_x = current_target_x - offset - 150
        new_y = current_target_y + 200 + (attempt * 40)
    elif attempt > 3:
        if dx > 0: new_x = current_target_x + offset + 80
        else: new_x = current_target_x - offset - 80
        new_y = current_target_y + 100 + (attempt * 40)
    else:
        if dx > 0: new_x = current_target_x + offset
        else: new_x = current_target_x - offset
        new_y = current_target_y + 30 + (attempt * 12)

    return new_x, new_y

# ========================================
# ИНИЦИАЛИЗАЦИЯ
# ========================================

print("\n" + "=" * 60)
print("🚁 СИСТЕМА МОНИТОРИНГА ПОЛЁТА БПЛА v14.3 (Высота + GPS)")
print("=" * 60 + "\n")

log = LogManager(log_dir="data/logs")
adapter = AirSimAdapter(log, use_simulation=False)

# СКАНЕР МИРА
world_scanner = WorldScanner(adapter.client, log)
world_scanner.scan_world()

# Выводим статистику
types = world_scanner.get_obstacle_types()
log.system("📊 Препятствия в мире:")
for obj_type, count in types.items():
    log.system(f"  {obj_type}: {count}")

# Сохраняем карту мира
world_scanner.save_to_file("data/world_objects.json")

obstacle = ObstacleDetector(log, world_scanner=world_scanner)
telemetry = TelemetryParser(log)
route = RoutePlanner(log)

# Цель
target_x, target_y = 200, 300
altitude = -5
speed = 12

# Генерируем данные для XAI
training_data, training_labels = generate_training_data(300)

feature_names = [
    'altitude', 'speed', 'obstacle_dist', 'battery',
    'obstacle_density', 'yaw', 'roll', 'pitch',
    'distance_to_target', 'angle_to_target'
]

# Обучаем модели
xai_model = train_xai_model(training_data, training_labels)
print("✅ XAI sklearn модель обучена")
torch_model = train_torch_model(training_data, training_labels)
print("✅ XAI torch модель обучена")

# XAI Bridge
xai = XAIBridge(log, training_data=training_data, feature_names=feature_names)
xai.set_model(xai_model)
xai.set_torch_model(torch_model)
xai_enabled = True

log.system("Запуск системы...")
log.system("Дрон: X-1000, версия 3.2")
log.system(f"Время старта: {datetime.now().strftime('%H:%M:%S')}")
log.system("XAI: АКТИВИРОВАН (авто-чередование: LIME/SHAP/LRP)")
log.system("Навигация: СКАНЕР МИРА + КОНТРОЛЬ ВЫСОТЫ")

route.plan(0, 0, target_x, target_y, altitude)

# ========================================
# ВЗЛЁТ
# ========================================

log.ok("Разрешение на взлёт получено")
if not adapter.takeoff():
    log.error("Не удалось взлететь!")
    sys.exit(1)

log.drone(f"🚁 Движение к цели: X={target_x} м, Y={target_y} м")
adapter.move_to(target_x, target_y, altitude, speed)

# ========================================
# ПАРАМЕТРЫ
# ========================================

MAX_FLIGHT_TIME = 600  # 10 минут

# 👈 ОГРАНИЧЕНИЕ ВЫСОТЫ
MAX_ALTITUDE = 10   # максимальная высота (метров)
MIN_ALTITUDE = 3    # минимальная высота (метров)

COLLISION_TIMEOUT = 10
collision_start_time = None
is_collision_mode = False
collision_attempts = 0
MAX_COLLISION_ATTEMPTS = 5
OBSTACLE_DETECTION_DISTANCE = 30.0

BASE_SPEED = 12
MIN_SPEED = 4
ADAPTIVE_SPEED = True
avoidance_side = 0

# Детекция тупика
MAX_STUCK_TIME = 10
position_history = []
stuck_time = 0
MAX_AVOID_ATTEMPTS = 5

# Счётчик попыток взлёта
takeoff_attempts = 0
MAX_TAKEOFF_ATTEMPTS = 5

# Память
obstacle_memory = {}
obstacle_avoided = set()
obstacle_attempts = {}
obstacle_avoid_count = {}
avoided_buildings = set()
MAX_AVOIDED_BUILDINGS = 10
danger_zones = []
danger_zones_visited = {}
MAX_DANGER_ZONE_VISITS = 3

# XAI статистика
xai_call_count = 0
xai_method_stats = {'lime': 0, 'shap': 0, 'lrp': 0, 'fallback': 0}

# Для проверки движения
last_position = None
move_check_counter = 0

# GPS данные
gps_data = None

# ========================================
# ОСНОВНОЙ ЦИКЛ
# ========================================

flight_time = 0
crash_detected = False
obstacle_count = 0
target_reached = False
xai_explanations = []

collision_stats = {
    'здание': 0, 'автобус': 0, 'дерево': 0, 'столб': 0,
    'машина': 0, 'человек': 0, 'светофор': 0, 'провод': 0,
    'неизвестно': 0
}

current_target_x = target_x
current_target_y = target_y
successful_avoidance = 0

try:
    while flight_time < MAX_FLIGHT_TIME:
        flight_time += 1

        # Получаем состояние
        state = adapter.get_state()
        if state is None:
            log.warn("⚠️ Состояние не получено, попытка переподключения...")
            time.sleep(0.5)
            continue

        pos = state['position']
        sensors = state.get('sensors', {})
        real_altitude = abs(pos['z'])

        # 👉 ПОЛУЧАЕМ GPS
        gps_data = adapter.get_gps_data()
        if gps_data and abs(gps_data['latitude']) > 0.01:
            log.drone(f"🌍 GPS: {gps_data['latitude']:.6f}, {gps_data['longitude']:.6f}")

        # 👈 КОНТРОЛЬ ВЫСОТЫ
        if real_altitude > MAX_ALTITUDE + 1:
            log.warn(f"⚠️ Слишком высоко! {real_altitude:.1f} м > {MAX_ALTITUDE} м")
            log.drone(f"📉 Снижаюсь до {MAX_ALTITUDE} м")
            adapter.move_to(pos['x'], pos['y'], -MAX_ALTITUDE, speed)
            time.sleep(0.5)
            continue

        if real_altitude < MIN_ALTITUDE and flight_time > 10:
            log.warn(f"⚠️ Слишком низко! {real_altitude:.1f} м < {MIN_ALTITUDE} м")
            log.drone(f"📈 Поднимаюсь до {MIN_ALTITUDE} м")
            adapter.move_to(pos['x'], pos['y'], -MIN_ALTITUDE, speed)
            time.sleep(0.5)
            continue

        # 👉 ПРОВЕРКА: ДРОН НА ЗЕМЛЕ?
        if real_altitude < 2.0 and flight_time > 5:
            takeoff_attempts += 1
            if takeoff_attempts <= MAX_TAKEOFF_ATTEMPTS:
                log.warn(f"⚠️ Дрон слишком низко ({real_altitude:.1f} м)! Попытка взлёта #{takeoff_attempts}...")
                try:
                    adapter.client.takeoffAsync()
                    time.sleep(2)
                    adapter.client.moveToPositionAsync(pos['x'], pos['y'], -MIN_ALTITUDE, 5.0, timeout_sec=3.0).join()
                    time.sleep(1)
                    continue
                except Exception as e:
                    log.error(f"❌ Ошибка принудительного взлёта: {e}")
                    continue
            else:
                log.error(f"❌ Не удалось взлететь после {MAX_TAKEOFF_ATTEMPTS} попыток!")
                crash_detected = True
                break

        # Проверка движения (каждые 5 секунд)
        if flight_time % 5 == 0 and flight_time > 10:
            if last_position is not None:
                dx = pos['x'] - last_position[0]
                dy = pos['y'] - last_position[1]
                dist_moved = (dx*dx + dy*dy) ** 0.5
                if dist_moved < 0.5:
                    log.warn(f"⚠️ Дрон не двигается! ({dist_moved:.2f} м за 5 сек)")
                    log.drone("🔄 Повторная отправка команды движения...")
                    adapter.move_to(target_x, target_y, -MIN_ALTITUDE, speed)
                else:
                    log.drone(f"✅ Дрон движется: {dist_moved:.2f} м за 5 сек")
            last_position = (pos['x'], pos['y'])

        # Адаптивная скорость
        if ADAPTIVE_SPEED:
            obstacle_dist = sensors.get('obstacle_distance', 100.0)
            if obstacle_dist < 15:
                current_speed = max(MIN_SPEED, speed * (obstacle_dist / 15))
            else:
                current_speed = speed
        else:
            current_speed = speed

        # ДЕТЕКЦИЯ ТУПИКА
        current_pos = (round(pos['x'], 1), round(pos['y'], 1))
        position_history.append(current_pos)
        if len(position_history) > 50:
            position_history.pop(0)

        if len(position_history) >= 10:
            unique_positions = len(set(position_history[-10:]))
            if unique_positions < 3:
                stuck_time += 1
                if stuck_time % 5 == 0:
                    log.warn(f"⏳ Дрон зациклился! {stuck_time} сек...")
                if stuck_time >= MAX_STUCK_TIME:
                    log.error("❌ Дрон зациклился! Кардинальная смена маршрута!")
                    new_x = pos['x'] + random.randint(50, 150) * (1 if random.random() > 0.5 else -1)
                    new_y = pos['y'] + random.randint(50, 150) * (1 if random.random() > 0.5 else -1)
                    adapter.move_to(new_x, new_y, -MIN_ALTITUDE, current_speed)
                    stuck_time = 0
                    position_history = []
                    continue
            else:
                stuck_time = max(0, stuck_time - 1)

        # Расстояние до цели
        distance_to_target = ((pos['x'] - target_x)**2 + (pos['y'] - target_y)**2) ** 0.5

        # Статус
        total_collisions = sum(collision_stats.values())
        if total_collisions == 0:
            log.drone("🟢 Лечу! Движение свободное")
        else:
            max_object = max(collision_stats, key=collision_stats.get)
            log.drone(f"🟡 Лечу! Впереди: {max_object.upper()} (коллизий: {collision_stats[max_object]})")

        # Телеметрия
        log.drone(f"📡 Высота: {real_altitude:.1f} м")
        log.drone(f"📡 До цели: {distance_to_target:.1f} м")

        # Проверка цели
        if distance_to_target < 2.0:
            target_reached = True
            log.ok(f"🎯 Цель достигнута! Обходов: {successful_avoidance}")
            break

        # ========================================
        # ДЕТЕКЦИЯ ПРЕПЯТСТВИЙ
        # ========================================

        detected = obstacle.detect(state)

        if detected:
            obstacle_count += 1
            object_name = detected.get('object', 'неизвестно')
            distance = detected.get('distance', 100.0)

            world_scanner.update_from_log(object_name)

            if object_name in collision_stats:
                collision_stats[object_name] += 1
            else:
                collision_stats['неизвестно'] += 1

            obstacle_key = (round(pos['x'], 1), round(pos['y'], 1))

            if obstacle_key in avoided_buildings:
                log.warn(f"⚠️ Здание {obstacle_key} уже обойдено, пропускаю!")
                continue

            if obstacle_key not in obstacle_memory:
                obstacle_memory[obstacle_key] = object_name
                log.info(f"📝 Запомнено препятствие: {object_name} в ({obstacle_key[0]}, {obstacle_key[1]})")
                obstacle_attempts[obstacle_key] = 0
                obstacle_avoid_count[obstacle_key] = 0

            if detected['type'] == 'collision':
                zone_exists = False
                for zone in danger_zones:
                    if abs(zone['x'] - pos['x']) < 5 and abs(zone['y'] - pos['y']) < 5:
                        zone_exists = True
                        break
                if not zone_exists:
                    danger_zones.append({
                        'x': pos['x'],
                        'y': pos['y'],
                        'object': object_name,
                        'time': flight_time
                    })
                    log.warn(f"⚠️ Добавлена опасная зона: {object_name} в ({pos['x']:.1f}, {pos['y']:.1f})")

            # ОБХОД
            if distance < OBSTACLE_DETECTION_DISTANCE and not is_collision_mode:
                obstacle_width = estimate_obstacle_size(sensors)
                obj_class = classify_object(distance, obstacle_width, 0)
                attempt = obstacle_attempts.get(obstacle_key, 0)
                avoid_count = obstacle_avoid_count.get(obstacle_key, 0)

                log.obstacle(f"⚠️ ПРЕПЯТСТВИЕ близко!")
                log.obstacle(f" 📏 Расстояние: {distance:.1f} м")
                log.obstacle(f" 📐 Размер: {obj_class['size_desc']} ({obstacle_width:.1f} м)")
                log.obstacle(f" 🔄 Попытка обхода #{attempt + 1}, уже обойдено: {avoid_count} раз")

                if attempt >= MAX_AVOID_ATTEMPTS:
                    log.warn(f"⚠️ Превышено число попыток обхода {object_name}!")
                    log.warn("➡️ КАРДИНАЛЬНАЯ СМЕНА МАРШРУТА!")
                    new_x = pos['x'] + random.randint(100, 300) * (1 if random.random() > 0.5 else -1)
                    new_y = pos['y'] + random.randint(100, 300) * (1 if random.random() > 0.5 else -1)
                    adapter.move_to(new_x, new_y, -MIN_ALTITUDE, current_speed)
                    obstacle_attempts[obstacle_key] = 0
                    time.sleep(0.3)
                    continue

                log.drone(f"🔄 Начинаю обход {object_name} (попытка {attempt+1})...")
                try:
                    collision_attempts += 1
                    offset_multiplier = 1 + (attempt * 0.5)
                    min_offset = max(obstacle_width / 2 + 40, 50) * offset_multiplier

                    if avoidance_side == 0:
                        offset_x = min_offset
                        offset_y = 30 + (attempt * 12)
                        avoidance_side = 1
                    else:
                        offset_x = -min_offset
                        offset_y = 30 + (attempt * 12)
                        avoidance_side = 0

                    # 👈 ОБХОД С ОГРАНИЧЕНИЕМ ВЫСОТЫ
                    current_z = -MIN_ALTITUDE
                    # Если препятствие большое, поднимаемся выше
                    if obstacle_width > 20:
                        current_z = -MAX_ALTITUDE
                        log.drone(f"📈 Большое препятствие! Поднимаюсь до {MAX_ALTITUDE} м")

                    adapter.move_to(pos['x'] + offset_x, pos['y'] + offset_y, current_z, current_speed)
                    time.sleep(0.3)

                    adapter.move_to(target_x, target_y, -MIN_ALTITUDE, current_speed)

                    obstacle_attempts[obstacle_key] = attempt + 1
                    obstacle_avoid_count[obstacle_key] = avoid_count + 1
                    obstacle_avoided.add(obstacle_key)
                    successful_avoidance += 1
                    avoided_buildings.add(obstacle_key)

                    if len(avoided_buildings) > MAX_AVOIDED_BUILDINGS:
                        avoided_buildings.pop()

                    log.info(f"🏷️ Здание {obstacle_key} добавлено в чёрный список")
                    log.ok(f"✅ Обход {object_name} выполнен! (всего: {successful_avoidance})")

                    is_collision_mode = False
                    collision_start_time = None

                except Exception as e:
                    log.error(f"❌ Ошибка обхода {object_name}: {e}")
                    crash_detected = True
                    break

        # ========================================
        # XAI ОБЪЯСНЕНИЕ
        # ========================================

        if xai_enabled and (flight_time % 10 == 0 or (detected and detected.get('type') in ['collision', 'large'])):
            try:
                if detected and detected.get('type') in ['collision', 'large']:
                    action = 0
                    confidence = 0.9
                else:
                    action = 1
                    confidence = 0.85

                xai_call_count += 1
                state['target_x'] = target_x
                state['target_y'] = target_y
                
                # 👈 ДОБАВЛЯЕМ GPS В СОСТОЯНИЕ ДЛЯ XAI
                if gps_data:
                    state['gps_lat'] = gps_data['latitude']
                    state['gps_lon'] = gps_data['longitude']
                    state['gps_alt'] = gps_data['altitude']

                explanation = xai.explain_decision(
                    state=state,
                    action=action,
                    confidence=confidence,
                    method="auto"
                )

                if explanation:
                    xai_explanations.append(explanation)
                    method_used = explanation.get('method', 'unknown')
                    xai_method_stats[method_used] = xai_method_stats.get(method_used, 0) + 1

                    if explanation.get('is_anomaly', {}).get('is_anomaly', False):
                        log.warn(f"📊 Аномалия: {explanation['is_anomaly']['reasons']}")
                    log.info(f"📊 Решение: {explanation.get('action_name')} (уверенность: {explanation.get('confidence', 0):.2f})")
                    log.info(f" 📊 Метод: {explanation.get('method')} ({explanation.get('runtime_ms', 0):.1f} мс)")

                if explanation and explanation.get('attribution') and flight_time % 30 == 0:
                    try:
                        import matplotlib.pyplot as plt
                        feature_names = explanation.get('feature_names', [])
                        attribution = np.array(explanation.get('attribution', []))
                        method_name = explanation.get('method', 'unknown')

                        if len(feature_names) == len(attribution):
                            plt.figure(figsize=(10, 6))
                            colors = ['green' if v > 0 else 'red' for v in attribution]
                            plt.barh(feature_names, attribution, color=colors, alpha=0.7)
                            plt.title(f'XAI Важность признаков ({method_name.upper()}, шаг {flight_time})')
                            plt.xlabel('Вклад в решение')
                            plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
                            plt.tight_layout()
                            plot_dir = "results/xai_plots"
                            os.makedirs(plot_dir, exist_ok=True)
                            plot_path = os.path.join(plot_dir, f"xai_{method_name}_step_{flight_time:03d}.png")
                            plt.savefig(plot_path, dpi=150)
                            plt.close()
                            log.info(f" 📊 График: {plot_path}")
                    except Exception as e:
                        log.warn(f" ⚠️ Ошибка визуализации: {e}")

            except Exception as e:
                log.error(f"❌ XAI ошибка: {e}")

        # ========================================
        # КООРДИНАТЫ
        # ========================================

        if flight_time % 5 == 0:
            log.drone(f"📍 Координаты: X={pos['x']:.1f}, Y={pos['y']:.1f}, Z={pos['z']:.1f}")
            log.drone(f"📍 Высота: {real_altitude:.1f} м")
            log.drone(f"🎯 До цели: {distance_to_target:.1f} м")

        time.sleep(0.2)

except KeyboardInterrupt:
    log.warn("⏹️ Полёт прерван пользователем (Ctrl+C)")
    print("\n" + "=" * 60)
    print("📊 СОХРАНЕНИЕ ДАННЫХ...")
    print("=" * 60)
    log.save()
    try:
        xai_path = f"data/logs/xai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        xai.save_explanations(xai_path)
        print(f"✅ XAI сохранены: {xai_path}")
    except:
        pass
    try:
        adapter.land()
    except:
        pass
    try:
        adapter.disconnect()
    except:
        pass
    print("\n✅ Аварийное завершение, данные сохранены!")
    print("=" * 60)
    sys.exit(0)

except Exception as e:
    log.error(f"❌ Критическая ошибка: {e}")
    crash_detected = True

# ========================================
# ПОСАДКА
# ========================================

if not crash_detected:
    log.drone("🛬 Начинаем посадку...")
    adapter.land()
else:
    log.error("🛑 АВАРИЙНАЯ ПОСАДКА!")
    adapter.land()

# ========================================
# ОТЧЁТ
# ========================================

print("\n" + "=" * 60)
print("📊 ИТОГОВЫЙ ОТЧЁТ")
print("=" * 60)

if crash_detected:
    log.error("СТАТУС: АВАРИЙНАЯ ОСТАНОВКА")
elif target_reached:
    log.ok(f"СТАТУС: ЦЕЛЬ ДОСТИГНУТА ✅ (обходов: {successful_avoidance})")
else:
    log.ok("СТАТУС: ПОЛЁТ ЗАВЕРШЁН")

log.info(f"⏱️ Длительность: {flight_time} сек")
log.info(f"🚧 Всего препятствий: {obstacle_count}")
log.info(f"🏆 Успешных обходов: {successful_avoidance}")

# 👈 GPS В ОТЧЁТЕ
if gps_data and gps_data['latitude'] != 0:
    log.info(f"🌍 GPS: {gps_data['latitude']:.6f}, {gps_data['longitude']:.6f}")
else:
    log.warn("⚠️ GPS данные отсутствуют")

# Статистика по объектам
print("\n" + "=" * 60)
print("📊 СТАТИСТИКА КОЛЛИЗИЙ ПО ОБЪЕКТАМ")
print("=" * 60)

total_collisions = sum(collision_stats.values())
for obj, count in collision_stats.items():
    if count > 0:
        pct = (count / total_collisions * 100) if total_collisions > 0 else 0
        log.info(f" {obj.upper()}: {count} ({pct:.1f}%)")

log.info(f"📊 Всего коллизий: {total_collisions}")

if total_collisions > 0:
    sorted_objects = sorted(collision_stats.items(), key=lambda x: x[1], reverse=True)
    print("\n🏆 ТОП-3 ОБЪЕКТА-ПРЕПЯТСТВИЯ:")
    for i, (obj, count) in enumerate(sorted_objects[:3], 1):
        if count > 0:
            log.info(f" {i}. {obj.upper()}: {count} раз")

# Запомненные препятствия
if obstacle_memory:
    print("\n" + "=" * 60)
    print("📝 ЗАПОМНЕННЫЕ ПРЕПЯТСТВИЯ")
    print("=" * 60)
    unique_obstacles = {}
    for (x, y), obj in obstacle_memory.items():
        if obj not in unique_obstacles:
            unique_obstacles[obj] = (x, y)
    for obj, (x, y) in list(unique_obstacles.items())[:5]:
        log.info(f" {obj} в ({x}, {y})")

# СТАТИСТИКА МИРА
print("\n" + "=" * 60)
print("📊 СТАТИСТИКА МИРА")
print("=" * 60)
types = world_scanner.get_obstacle_types()
for obj_type, count in types.items():
    log.info(f" {obj_type}: {count} объектов")

# XAI ОТЧЁТ
print("\n" + "=" * 60)
print("📊 XAI ОТЧЁТ")
print("=" * 60)

if xai_enabled:
    xai_summary = xai.get_summary()
    log.info(f"Всего объяснений: {xai_summary.get('count', 0)}")
    log.info(f"Аномалий: {xai_summary.get('anomaly_count', 0)}")
    log.info(f"Доля: {xai_summary.get('anomaly_rate', 0):.1%}")
    log.info(f"Использованные методы: {xai_summary.get('methods_used', [])}")

    print("\n" + "=" * 60)
    print("📊 СТАТИСТИКА XAI МЕТОДОВ")
    print("=" * 60)

    for method, count in xai_method_stats.items():
        if count > 0:
            pct = (count / xai_call_count * 100) if xai_call_count > 0 else 0
            log.info(f" {method.upper()}: {count} ({pct:.1f}%)")

    xai_path = f"data/logs/xai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    xai.save_explanations(xai_path)
    log.info(f"XAI сохранены: {xai_path}")

log.save()
adapter.disconnect()

print("\n✅ Мониторинг завершён!")