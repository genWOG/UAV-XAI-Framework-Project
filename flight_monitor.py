"""
Главный модуль мониторинга полёта с XAI интеграцией
Версия 10.1 — исправлено сохранение XAI объяснений
"""

import sys
import os
import time
import json
import math
import numpy as np
from datetime import datetime

# Добавляем пути
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'xai', 'modules'))

from modules.log_manager import LogManager
from modules.obstacle_detector import ObstacleDetector
from modules.telemetry_parser import TelemetryParser
from modules.route_planner import RoutePlanner
from modules.airsim_adapter import AirSimAdapter

# XAI Bridge
from xai.modules.xai_bridge import XAIBridge
XAI_AVAILABLE = True
print("✅ XAI Bridge загружен")


# ========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ========================================

def generate_training_data(n_samples=300):
    """Генерирует синтетические данные для обучения XAI-модели."""
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
        
        # Логика решения: остановиться если близко препятствие или низкая батарея
        if obstacle_dist < 10 or battery < 15:
            label = 0  # Остановка
        else:
            label = 1  # Продолжить
            
        data.append([altitude, speed, obstacle_dist, battery, density, yaw, roll, pitch])
        labels.append(label)
    
    return np.array(data), np.array(labels)


def train_xai_model(X, y):
    """Обучает простую модель для XAI."""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model


def train_torch_model(X, y):
    """Обучает PyTorch модель для LRP."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    class SimpleNN(nn.Module):
        def __init__(self, input_dim=8, hidden_dim=16, output_dim=2):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleNN()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Преобразуем в тензоры
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Обучаем
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model


def estimate_obstacle_size(sensors):
    """Оценивает размеры препятствия по данным сенсоров."""
    distance = sensors.get('obstacle_distance', 100.0)
    density = sensors.get('obstacle_density', 0.0)
    if distance < 5:
        return 50
    elif distance < 15:
        return 30
    elif distance < 30:
        return 20
    elif density > 0.8:
        return 50
    elif density > 0.5:
        return 25
    else:
        return 15


def classify_object(distance, width, speed):
    """Классифицирует объект по размеру и скорости."""
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
        'size': size,
        'size_desc': size_desc,
        'motion': motion,
        'motion_desc': motion_desc,
        'full_desc': f"{size_desc} ({size}), {motion_desc}"
    }


def calculate_new_target(pos, obstacle_width, current_target_x, current_target_y, attempt=0):
    """Рассчитывает новую цель после обхода препятствия."""
    dx = current_target_x - pos['x']
    dy = current_target_y - pos['y']
    base_offset = max(obstacle_width / 2 + 40, 50)
    offset = base_offset + (attempt * 25)
    
    if attempt > 5:
        if dx > 0:
            new_x = current_target_x + offset + 150
        else:
            new_x = current_target_x - offset - 150
        new_y = current_target_y + 200 + (attempt * 40)
    elif attempt > 3:
        if dx > 0:
            new_x = current_target_x + offset + 80
        else:
            new_x = current_target_x - offset - 80
        new_y = current_target_y + 100 + (attempt * 40)
    else:
        if dx > 0:
            new_x = current_target_x + offset
        else:
            new_x = current_target_x - offset
        new_y = current_target_y + 30 + (attempt * 12)
    
    return new_x, new_y


# ========================================
# ИНИЦИАЛИЗАЦИЯ
# ========================================

print("\n" + "="*60)
print("🚁 СИСТЕМА МОНИТОРИНГА ПОЛЁТА БПЛА v10.1 (XAI Auto + Исправлено)")
print("="*60 + "\n")

log = LogManager(log_dir="data/logs")
adapter = AirSimAdapter(log, use_simulation=False)
obstacle = ObstacleDetector(log)
telemetry = TelemetryParser(log)
route = RoutePlanner(log)

# 👉 ГЕНЕРИРУЕМ ДАННЫЕ ДЛЯ XAI
training_data, training_labels = generate_training_data(300)
feature_names = ['altitude', 'speed', 'obstacle_dist', 'battery',
                 'obstacle_density', 'yaw', 'roll', 'pitch']

# 👉 ОБУЧАЕМ МОДЕЛЬ ДЛЯ LIME/SHAP
xai_model = train_xai_model(training_data, training_labels)
print("✅ XAI sklearn модель обучена")

# 👉 ОБУЧАЕМ TORCH МОДЕЛЬ ДЛЯ LRP
torch_model = train_torch_model(training_data, training_labels)
print("✅ XAI torch модель обучена")

# 👉 ИНИЦИАЛИЗИРУЕМ XAI BRIDGE
xai = XAIBridge(log, training_data=training_data, feature_names=feature_names)
xai.set_model(xai_model)
xai.set_torch_model(torch_model)
xai_enabled = True

log.system("Запуск системы...")
log.system(f"Режим: {'РЕАЛЬНЫЙ' if not adapter.use_simulation else 'СИМУЛЯЦИЯ'}")
log.system("Дрон: X-1000, версия 3.2")
log.system(f"Время старта: {datetime.now().strftime('%H:%M:%S')}")
log.system(f"XAI: АКТИВИРОВАН (авто-чередование: LIME/SHAP/LRP)")

# Цель
target_x, target_y = 200, 300
altitude = -5
speed = 12
route.plan(0, 0, target_x, target_y, altitude)

# ========================================
# ВЗЛЁТ
# ========================================

log.ok("Разрешение на взлёт получено")
if not adapter.takeoff():
    log.error("Не удалось взлететь!")
    sys.exit(1)

# ========================================
# ДВИЖЕНИЕ К ЦЕЛИ
# ========================================

log.drone(f"🚁 Движение к цели: X={target_x} м, Y={target_y} м")
adapter.move_to(target_x, target_y, altitude, speed)

# ========================================
# ПАРАМЕТРЫ
# ========================================

COLLISION_TIMEOUT = 10
collision_start_time = None
is_collision_mode = False
collision_attempts = 0
MAX_COLLISION_ATTEMPTS = 5
OBSTACLE_DETECTION_DISTANCE = 30.0
MAX_FLIGHT_TIME = 600

avoidance_side = 0
obstacle_memory = {}
obstacle_avoided = set()
obstacle_attempts = {}
obstacle_avoid_count = {}
avoided_buildings = set()
MAX_AVOIDED_BUILDINGS = 10
danger_zones = []
danger_zones_visited = {}
MAX_DANGER_ZONE_VISITS = 3

# 👈 СЧЁТЧИКИ XAI
xai_call_count = 0
xai_method_stats = {'lime': 0, 'shap': 0, 'lrp': 0, 'fallback': 0}

# ========================================
# ОСНОВНОЙ ЦИКЛ
# ========================================

flight_time = 0
crash_detected = False
obstacle_count = 0
target_reached = False
xai_explanations = []  # 👈 БУДЕТ ЗАПОЛНЯТЬСЯ

collision_stats = {
    'здание': 0, 'автобус': 0, 'дерево': 0, 'столб': 0,
    'машина': 0, 'человек': 0, 'светофор': 0, 'провод': 0,
    'неизвестно': 0
}

stuck_time = 0
last_position = None
current_target_x = target_x
current_target_y = target_y
successful_avoidance = 0
prev_distance_to_target = None
backward_counter = 0

try:
    while flight_time < MAX_FLIGHT_TIME:
        flight_time += 1
        
        state = adapter.get_state()
        pos = state['position']
        sensors = state.get('sensors', {})
        real_altitude = abs(pos['z'])
        
        # 👉 ПРОВЕРКА ЗАВИСАНИЯ
        current_pos = (round(pos['x'], 1), round(pos['y'], 1), round(pos['z'], 1))
        if last_position is not None:
            if abs(current_pos[0] - last_position[0]) < 0.2 and abs(current_pos[1] - last_position[1]) < 0.2:
                if abs(current_pos[2] - last_position[2]) < 0.2:
                    stuck_time += 1
                    if stuck_time % 5 == 0:
                        log.warn(f"⏳ Дрон стоит на месте {stuck_time} сек...")
                    if stuck_time >= 15:
                        log.error(f"❌ Дрон завис на 15 сек! Аварийная остановка")
                        crash_detected = True
                        break
                else:
                    if stuck_time > 0:
                        log.warn("🔄 Дрон раскачивается, даю шанс...")
                        stuck_time = max(0, stuck_time - 2)
            else:
                if stuck_time > 0:
                    log.ok(f"✅ Дрон снова двигается (стоял {stuck_time} сек)")
                    stuck_time = 0
            last_position = current_pos
        else:
            last_position = current_pos
        
        # 👉 СТАТУС
        total_collisions = sum(collision_stats.values())
        if total_collisions == 0:
            log.drone("🟢 Лечу! Движение свободное")
        else:
            max_object = max(collision_stats, key=collision_stats.get)
            if collision_stats[max_object] > 0:
                log.drone(f"🟡 Лечу! Впереди: {max_object.upper()} (коллизий: {collision_stats[max_object]})")
        
        # 👉 ТЕЛЕМЕТРИЯ
        gps = sensors.get('gps', {})
        log.drone(f"📡 GPS: {gps.get('lat', 0):.6f}, {gps.get('lon', 0):.6f}")
        log.drone(f"📡 Высота: {real_altitude:.1f} м")
        
        distance_to_target = ((pos['x'] - current_target_x)**2 + (pos['y'] - current_target_y)**2) ** 0.5
        
        # 👉 ОГРАНИЧЕНИЕ ВЫСОТЫ
        if real_altitude > adapter.max_altitude + 2:
            log.warn(f"⚠️ Слишком высоко! {real_altitude:.1f} м, снижаюсь до {adapter.max_altitude} м")
            adapter.move_to(pos['x'], pos['y'], -adapter.max_altitude, speed, force=True)
            time.sleep(0.3)
            continue
        
        # 👉 ПРОВЕРКА ДВИЖЕНИЯ НАЗАД
        if flight_time > 10 and not is_collision_mode:
            if prev_distance_to_target is not None:
                if distance_to_target > prev_distance_to_target + 2:
                    backward_counter += 1
                    if backward_counter >= 3:
                        log.warn(f"⚠️ Дрон движется назад! Расстояние до цели: {distance_to_target:.1f} м")
                        log.drone(f"🔄 Принудительное перенаправление к цели ({target_x}, {target_y})")
                        adapter.move_to(target_x, target_y, altitude, speed + 3, force=True)
                        backward_counter = 0
                        time.sleep(0.3)
                        continue
                else:
                    backward_counter = max(0, backward_counter - 1)
            prev_distance_to_target = distance_to_target
        
        # 👉 КОРРЕКЦИЯ МАРШРУТА (каждые 30 сек)
        if distance_to_target > 30 and flight_time > 10 and not is_collision_mode and flight_time % 30 == 0:
            log.drone(f"🔄 Коррекция маршрута: цель ({current_target_x:.0f}, {current_target_y:.0f})")
            adapter.move_to(current_target_x, current_target_y, altitude, speed, force=True)
            time.sleep(0.3)
        
        # 👉 ПРОВЕРКА ДОСТИЖЕНИЯ ЦЕЛИ
        if distance_to_target < 2.0 and not target_reached:
            if abs(current_target_x - target_x) > 0.1 or abs(current_target_y - target_y) > 0.1:
                log.drone(f"📍 Достигнута промежуточная цель! Двигаюсь к основной: ({target_x}, {target_y})")
                current_target_x = target_x
                current_target_y = target_y
                adapter.move_to(current_target_x, current_target_y, altitude, speed, force=True)
                continue
            else:
                target_reached = True
                log.ok(f"🎯 Цель достигнута! Обходов: {successful_avoidance}")
                break
        
        # 👉 ПРОВЕРКА ОПАСНЫХ ЗОН
        danger_detected = False
        for danger_zone in danger_zones:
            dx = pos['x'] - danger_zone['x']
            dy = pos['y'] - danger_zone['y']
            if abs(dx) < 25 and abs(dy) < 25 and not is_collision_mode:
                zone_key = (round(danger_zone['x'], 1), round(danger_zone['y'], 1))
                
                if zone_key in avoided_buildings:
                    log.warn(f"⚠️ Зона {zone_key} уже обойдена, пропускаю!")
                    current_target_x = target_x
                    current_target_y = target_y
                    adapter.move_to(current_target_x, current_target_y, altitude, speed, force=True)
                    continue
                
                visits = danger_zones_visited.get(zone_key, 0)
                if visits >= MAX_DANGER_ZONE_VISITS:
                    log.warn(f"⚠️ Зона {zone_key} уже обойдена {visits} раз, пытаюсь прорваться!")
                    try:
                        is_collision_mode = False
                        collision_start_time = None
                        log.drone(f"💨 Прорыв к цели! ({target_x}, {target_y})")
                        adapter.move_to(target_x, target_y, altitude, speed + 3, force=True)
                        time.sleep(0.3)
                        new_state = adapter.get_state()
                        new_pos = new_state['position']
                        if abs(new_pos['x'] - pos['x']) > 1 or abs(new_pos['y'] - pos['y']) > 1:
                            log.ok("✅ Прорыв удался! Двигаюсь дальше")
                            danger_zones = [z for z in danger_zones if z != danger_zone]
                            danger_zones_visited[zone_key] = 0
                            avoided_buildings.add(zone_key)
                            danger_detected = True
                            break
                        else:
                            log.warn("⚠️ Прорыв не удался, пробую обход с большим радиусом")
                            new_x, new_y = calculate_new_target(pos, 50, current_target_x, current_target_y, 10)
                            log.drone(f"🔄 Новая цель: ({new_x:.0f}, {new_y:.0f})")
                            current_target_x = new_x
                            current_target_y = new_y
                            adapter.move_to(current_target_x, current_target_y, altitude, speed, force=True)
                            danger_detected = True
                            break
                    except Exception as e:
                        log.error(f"❌ Ошибка прорыва: {e}")
                        continue
                
                danger_detected = True
                log.warn(f"🔮 Предсказано препятствие в опасной зоне! (обход #{visits+1})")
                log.warn(f"➡️ Заблаговременный обход!")
                
                try:
                    obstacle_width = 50
                    attempt = obstacle_attempts.get('danger_zone', 0)
                    min_offset = max(obstacle_width / 2 + 40, 50) + (attempt * 25)
                    
                    if avoidance_side == 0:
                        offset_x = min_offset
                        offset_y = 30 + (attempt * 12)
                        avoidance_side = 1
                    else:
                        offset_x = -min_offset
                        offset_y = 30 + (attempt * 12)
                        avoidance_side = 0
                    
                    adapter.move_to(pos['x'] + offset_x, pos['y'] + offset_y, altitude, speed)
                    time.sleep(0.3)
                    
                    new_x, new_y = calculate_new_target(pos, obstacle_width, current_target_x, current_target_y, attempt + 1)
                    current_target_x = new_x
                    current_target_y = new_y
                    adapter.move_to(current_target_x, current_target_y, altitude, speed, force=True)
                    
                    obstacle_attempts['danger_zone'] = attempt + 1
                    danger_zones_visited[zone_key] = visits + 1
                    successful_avoidance += 1
                    avoided_buildings.add(zone_key)
                    if len(avoided_buildings) > MAX_AVOIDED_BUILDINGS:
                        avoided_buildings.pop()
                    
                    log.ok(f"✅ Предсказанный обход выполнен! (всего: {successful_avoidance})")
                except Exception as e:
                    log.error(f"❌ Ошибка предсказанного обхода: {e}")
                    break
                
        if danger_detected:
            continue
        
        # 👉 ДЕТЕКЦИЯ ПРЕПЯТСТВИЙ
        detected = obstacle.detect(state)
        collision_detected = False
        
        if detected:
            obstacle_count += 1
            object_name = detected.get('object', 'неизвестно')
            distance = detected.get('distance', 100.0)
            
            if object_name in collision_stats:
                collision_stats[object_name] += 1
            else:
                collision_stats['неизвестно'] += 1
            
            obstacle_key = (round(pos['x'], 1), round(pos['y'], 1))
            
            # Проверяем чёрный список
            if obstacle_key in avoided_buildings:
                log.warn(f"⚠️ Здание {obstacle_key} уже обойдено, пропускаю!")
                current_target_x = target_x
                current_target_y = target_y
                adapter.move_to(current_target_x, current_target_y, altitude, speed, force=True)
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
            
            # 👉 ОБХОД
            if distance < OBSTACLE_DETECTION_DISTANCE and not is_collision_mode:
                obstacle_width = estimate_obstacle_size(sensors)
                obj_class = classify_object(distance, obstacle_width, 0)
                attempt = obstacle_attempts.get(obstacle_key, 0)
                avoid_count = obstacle_avoid_count.get(obstacle_key, 0)
                
                log.obstacle(f"⚠️ ПРЕПЯТСТВИЕ близко!")
                log.obstacle(f" 📏 Расстояние: {distance:.1f} м")
                log.obstacle(f" 📐 Размер: {obj_class['size_desc']} ({obstacle_width:.1f} м)")
                log.obstacle(f" 🔄 Попытка обхода #{attempt + 1}, уже обойдено: {avoid_count} раз")
                
                if attempt >= 3:
                    log.warn(f"⚠️ Слишком много попыток обхода {object_name}!")
                    log.warn(f"➡️ Кардинальная смена маршрута!")
                    new_x, new_y = calculate_new_target(pos, obstacle_width, current_target_x, current_target_y, attempt + 5)
                    log.drone(f"🔄 Новая цель: ({new_x:.0f}, {new_y:.0f})")
                    current_target_x = new_x
                    current_target_y = new_y
                    adapter.move_to(current_target_x, current_target_y, altitude, speed, force=True)
                    obstacle_attempts[obstacle_key] = attempt + 1
                    time.sleep(0.3)
                    continue
                
                log.drone(f"🔄 Начинаю обход {object_name} (попытка {attempt+1})...")
                
                try:
                    collision_attempts += 1
                    min_offset = max(obstacle_width / 2 + 40, 50) + (attempt * 25)
                    
                    if avoidance_side == 0:
                        offset_x = min_offset
                        offset_y = 30 + (attempt * 12)
                        log.drone(f" ➡️ Обход СПРАВА (смещение {offset_x:.0f} м)")
                        avoidance_side = 1
                    else:
                        offset_x = -min_offset
                        offset_y = 30 + (attempt * 12)
                        log.drone(f" ⬅️ Обход СЛЕВА (смещение {offset_x:.0f} м)")
                        avoidance_side = 0
                    
                    log.drone(f" ✈️ Облёт препятствия...")
                    adapter.move_to(pos['x'] + offset_x, pos['y'] + offset_y, altitude, speed)
                    time.sleep(0.3)
                    
                    new_x, new_y = calculate_new_target(pos, obstacle_width, current_target_x, current_target_y, attempt + 1)
                    log.drone(f" 🎯 Новая цель: ({new_x:.0f}, {new_y:.0f})")
                    current_target_x = new_x
                    current_target_y = new_y
                    adapter.move_to(current_target_x, current_target_y, altitude, speed, force=True)
                    
                    obstacle_attempts[obstacle_key] = attempt + 1
                    obstacle_avoid_count[obstacle_key] = avoid_count + 1
                    obstacle_avoided.add(obstacle_key)
                    successful_avoidance += 1
                    
                    # Добавляем в чёрный список
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
                continue
        
        # ========================================
        # 👉 XAI ОБЪЯСНЕНИЕ (авто-чередование)
        # ========================================
        
        if xai_enabled and (flight_time % 10 == 0 or (detected and detected.get('type') in ['collision', 'large'])):
            try:
                # Определяем действие
                if detected and detected.get('type') in ['collision', 'large']:
                    action = 0  # Остановка
                    confidence = 0.9
                else:
                    action = 1  # Продолжить
                    confidence = 0.85
                
                xai_call_count += 1
                
                # 👈 ВЫЗЫВАЕМ XAI С АВТО-ЧЕРЕДОВАНИЕМ
                explanation = xai.explain_decision(
                    state=state,
                    action=action,
                    confidence=confidence,
                    method="auto"
                )
                
                if explanation:
                    # 👈 ДОБАВЛЯЕМ В ЛОКАЛЬНЫЙ СПИСОК (ДЛЯ ОТЧЁТА)
                    xai_explanations.append(explanation)
                    
                    # Считаем статистику по методам
                    method_used = explanation.get('method', 'unknown')
                    if method_used in xai_method_stats:
                        xai_method_stats[method_used] += 1
                    else:
                        xai_method_stats[method_used] = 1
                    
                    # Логируем
                    if explanation.get('is_anomaly', {}).get('is_anomaly', False):
                        log.warn(f"📊 Аномалия: {explanation['is_anomaly']['reasons']}")
                    
                    log.info(f"📊 Решение: {explanation.get('action_name')} (уверенность: {explanation.get('confidence', 0):.2f})")
                    log.info(f" 📊 Метод: {explanation.get('method')} ({explanation.get('runtime_ms', 0):.1f} мс)")
                    
                    # 👉 СОХРАНЯЕМ ГРАФИК (каждые 30 секунд)
                    if explanation.get('attribution') and flight_time % 30 == 0:
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
        # КООРДИНАТЫ (каждые 5 секунд)
        # ========================================
        
        if flight_time % 5 == 0:
            log.drone(f"📍 Координаты: X={pos['x']:.1f}, Y={pos['y']:.1f}, Z={pos['z']:.1f}")
            log.drone(f"📍 Высота: {real_altitude:.1f} м")
            log.drone(f"🎯 До цели: {distance_to_target:.1f} м")
        
        time.sleep(0.2)

except KeyboardInterrupt:
    log.warn("⏹️ Полёт прерван пользователем")

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

print("\n" + "="*60)
print("📊 ИТОГОВЫЙ ОТЧЁТ")
print("="*60)

if crash_detected:
    log.error("СТАТУС: АВАРИЙНАЯ ОСТАНОВКА")
elif target_reached:
    log.ok(f"СТАТУС: ЦЕЛЬ ДОСТИГНУТА ✅ (обходов: {successful_avoidance})")
else:
    log.ok("СТАТУС: ПОЛЁТ ЗАВЕРШЁН")

log.info(f"⏱️ Длительность: {flight_time} сек")
log.info(f"🚧 Всего препятствий: {obstacle_count}")
log.info(f"📝 Запомнено препятствий: {len(obstacle_memory)}")
log.info(f"🔄 Обойдено препятствий: {len(obstacle_avoided)}")
log.info(f"🏆 Успешных обходов: {successful_avoidance}")
log.info(f"🏷️ В чёрном списке: {len(avoided_buildings)} зданий")

# 👉 СТАТИСТИКА ПО ОБЪЕКТАМ
print("\n" + "="*60)
print("📊 СТАТИСТИКА КОЛЛИЗИЙ ПО ОБЪЕКТАМ")
print("="*60)

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

# 👉 ЗАПОМНЕННЫЕ ПРЕПЯТСТВИЯ
if obstacle_memory:
    print("\n" + "="*60)
    print("📝 ЗАПОМНЕННЫЕ ПРЕПЯТСТВИЯ")
    print("="*60)
    unique_obstacles = {}
    for (x, y), obj in obstacle_memory.items():
        if obj not in unique_obstacles:
            unique_obstacles[obj] = (x, y)
    for obj, (x, y) in list(unique_obstacles.items())[:5]:
        log.info(f" {obj} в ({x}, {y})")

# 👉 XAI ОТЧЁТ
print("\n" + "="*60)
print("📊 XAI ОТЧЁТ")
print("="*60)

if xai_enabled:
    # Получаем статистику из XAI Bridge
    xai_summary = xai.get_summary()
    
    log.info(f"Всего объяснений: {xai_summary.get('count', 0)}")
    log.info(f"Аномалий: {xai_summary.get('anomaly_count', 0)}")
    log.info(f"Доля: {xai_summary.get('anomaly_rate', 0):.1%}")
    log.info(f"Использованные методы: {xai_summary.get('methods_used', [])}")
    
    # 👉 СТАТИСТИКА ПО МЕТОДАМ
    print("\n" + "="*60)
    print("📊 СТАТИСТИКА XAI МЕТОДОВ")
    print("="*60)
    for method, count in xai_method_stats.items():
        if count > 0:
            pct = (count / xai_call_count * 100) if xai_call_count > 0 else 0
            log.info(f" {method.upper()}: {count} ({pct:.1f}%)")
    
    # 👉 СОХРАНЯЕМ XAI ОБЪЯСНЕНИЯ
    xai_path = f"data/logs/xai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    xai.save_explanations(xai_path)
    log.info(f"XAI сохранены: {xai_path}")
    
    # 👉 СОХРАНЯЕМ ЛОКАЛЬНУЮ КОПИЮ (на всякий случай)
    if xai_explanations:
        local_path = f"data/logs/xai_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(local_path, 'w', encoding='utf-8') as f:
                json.dump(xai_explanations, f, indent=2, default=str, ensure_ascii=False)
            log.info(f"XAI локально: {local_path}")
        except Exception as e:
            log.warn(f"⚠️ Ошибка сохранения локальной копии: {e}")
else:
    log.info("XAI не включён")

log.save()
adapter.disconnect()

print("\n✅ Мониторинг завершён!")