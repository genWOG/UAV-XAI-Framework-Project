"""
Главный модуль мониторинга полёта с XAI интеграцией
"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'xai', 'modules'))

from modules.log_manager import LogManager
from modules.obstacle_detector import ObstacleDetector
from modules.telemetry_parser import TelemetryParser
from modules.route_planner import RoutePlanner
from modules.airsim_adapter import AirSimAdapter

# XAI Bridge
try:
    from xai.modules.xai_bridge import XAIBridge
    XAI_AVAILABLE = True
    print("✅ XAI Bridge загружен")
except ImportError as e:
    XAI_AVAILABLE = False
    print(f"⚠️ XAI Bridge не найден: {e}")


print("\n" + "="*60)
print("🚁 СИСТЕМА МОНИТОРИНГА ПОЛЁТА БПЛА v4.0 (XAI)")
print("="*60 + "\n")

log = LogManager(log_dir="data/logs")
adapter = AirSimAdapter(log, use_simulation=False)
obstacle = ObstacleDetector(log)
telemetry = TelemetryParser(log)
route = RoutePlanner(log)

if XAI_AVAILABLE:
    xai = XAIBridge(log)
    xai_enabled = True
else:
    xai = None
    xai_enabled = False

log.system("Запуск системы...")
log.system(f"Режим: {'РЕАЛЬНЫЙ' if not adapter.use_simulation else 'СИМУЛЯЦИЯ'}")
log.system("Дрон: X-1000, версия 3.2")
log.system(f"Время старта: {datetime.now().strftime('%H:%M:%S')}")
log.system(f"XAI: {'АКТИВИРОВАН' if xai_enabled else 'ОТКЛЮЧЕН'}")

target_x, target_y = 50, 100
altitude = -1
speed = 3
route.plan(0, 0, target_x, target_y, altitude)

log.ok("Разрешение на взлёт получено")
if not adapter.takeoff():
    log.error("Не удалось взлететь!")
    sys.exit(1)

log.drone(f"🚁 Движение к цели: X={target_x} м, Y={target_y} м")
adapter.move_to(target_x, target_y, altitude, speed)

# 👉 Ждём начала движения (макс 10 сек)
timeout = 10
start_wait = time.time()
motion_started = False
while time.time() - start_wait < timeout:
    state = adapter.get_state()
    pos = state['position']
    if abs(pos['x']) > 0.1 or abs(pos['y']) > 0.1:
        log.ok("✅ Дрон начал движение")
        motion_started = True
        break
    time.sleep(0.5)

if not motion_started:
    log.error("❌ Дрон не двигается! Проверьте симулятор")
    # Продолжаем выполнение, но с предупреждением

# ========================================
# ПАРАМЕТРЫ
# ========================================
COLLISION_TIMEOUT = 10
collision_start_time = None
is_collision_mode = False
collision_attempts = 0
MAX_COLLISION_ATTEMPTS = 3
MAX_COLLISION_COUNT = 300  # Лимит коллизий

# ========================================
# ОСНОВНОЙ ЦИКЛ
# ========================================
flight_time = 0
crash_detected = False
obstacle_count = 0
max_flight_time = 60
target_reached = False
xai_explanations = []

# Словарь для подсчёта коллизий по типам объектов
collision_stats = {
    'здание': 0,
    'автобус': 0,
    'дерево': 0,
    'столб': 0,
    'машина': 0,
    'человек': 0,
    'неизвестно': 0
}

# Список возможных объектов (из AirSim)
OBJECT_TYPES = {
    'Building': 'здание',
    'Vehicle': 'автобус',
    'Tree': 'дерево',
    'Pole': 'столб',
    'Car': 'машина',
    'Human': 'человек',
    'Unknown': 'неизвестно'
}


def get_object_name(object_data):
    """Определяет тип объекта по данным из AirSim"""
    if object_data is None:
        return 'неизвестно'
    
    if isinstance(object_data, str):
        for eng, rus in OBJECT_TYPES.items():
            if eng.lower() in object_data.lower():
                return rus
        return 'неизвестно'
    
    if isinstance(object_data, dict):
        name = object_data.get('name', '')
        for eng, rus in OBJECT_TYPES.items():
            if eng.lower() in name.lower():
                return rus
        return 'неизвестно'
    
    return 'неизвестно'


try:
    while flight_time < max_flight_time:
        flight_time += 1
        state = adapter.get_state()
        pos = state['position']
        
        # 👉 СТАТУС КАЖДУЮ СЕКУНДУ
        total_collisions = sum(collision_stats.values())
        if total_collisions == 0:
            log.drone("🟢 Лечу! Движение свободное")
        else:
            max_object = max(collision_stats, key=collision_stats.get)
            if collision_stats[max_object] > 0:
                log.drone(f"🟡 Лечу! Впереди: {max_object.upper()} (коллизий: {collision_stats[max_object]})")
            else:
                log.drone("🟢 Лечу! Движение свободное")
        
        distance_to_target = ((pos['x'] - target_x)**2 + (pos['y'] - target_y)**2) ** 0.5
        if distance_to_target < 2.0 and not target_reached:
            target_reached = True
            log.ok(f"🎯 Цель достигнута!")
            break
        
        # 👉 ДЕТЕКЦИЯ ПРЕПЯТСТВИЙ
        detected = obstacle.detect(state)
        collision_detected = False
        
        if detected:
            obstacle_count += 1
            
            object_data = detected.get('object', None)
            object_name = get_object_name(object_data)
            
            if object_name in collision_stats:
                collision_stats[object_name] += 1
            else:
                collision_stats['неизвестно'] += 1
            
            # 👉 ВЫВОДИМ ЧТО ВПЕРЕДИ
            log.obstacle(f"⚠️ ВПЕРЕДИ: {object_name.upper()}! Расстояние: {detected.get('distance', 0):.1f} м")
            
            # Проверяем лимит коллизий
            total_collisions = sum(collision_stats.values())
            if total_collisions >= MAX_COLLISION_COUNT:
                log.error(f"🚨 ДОСТИГНУТ ЛИМИТ КОЛЛИЗИЙ: {total_collisions} из {MAX_COLLISION_COUNT}")
                crash_detected = True
                break
            
            if detected['type'] == 'collision':
                log.error(f"🚨 СТОЛКНОВЕНИЕ с {object_name}!")
                if not is_collision_mode:
                    is_collision_mode = True
                    collision_start_time = time.time()
                    collision_attempts += 1
                
                collision_detected = True
                elapsed = time.time() - collision_start_time
                if elapsed > COLLISION_TIMEOUT:
                    log.error(f"❌ Таймаут {COLLISION_TIMEOUT} сек! Не удалось обойти {object_name}")
                    crash_detected = True
                    break
                
                if collision_attempts > MAX_COLLISION_ATTEMPTS:
                    log.error(f"❌ Превышено число попыток обхода {object_name}")
                    crash_detected = True
                    break
                
                try:
                    log.drone(f"🔄 Обход {object_name}: попытка {collision_attempts}")
                    adapter.move_to(pos['x'], pos['y'], -4, speed)
                    time.sleep(1)
                    offset_x = 10 if collision_attempts % 2 == 1 else -10
                    offset_y = 10 if collision_attempts % 2 == 0 else -10
                    adapter.move_to(pos['x'] + offset_x, pos['y'] + offset_y, -4, speed)
                    time.sleep(1)
                    adapter.move_to(pos['x'] + offset_x, pos['y'] + offset_y, altitude, speed)
                    time.sleep(1)
                    adapter.move_to(target_x, target_y, altitude, speed)
                    log.ok(f"✅ Обход {object_name} выполнен")
                    is_collision_mode = False
                    collision_start_time = None
                except Exception as e:
                    log.error(f"❌ Ошибка обхода {object_name}: {e}")
                    crash_detected = True
                    break
                continue
            
            elif detected['type'] == 'large':
                log.obstacle(f"⚠️ Опасное {object_name} на расстоянии {detected['distance']:.1f} м!")
                if detected['distance'] < 5:
                    try:
                        offset = 15 if collision_attempts % 2 == 0 else -15
                        log.drone(f"🔄 Обход {object_name}: отворот")
                        adapter.move_to(pos['x'] + offset, pos['y'] + 10, altitude, speed)
                        time.sleep(1)
                        adapter.move_to(target_x, target_y, altitude, speed)
                        log.ok(f"✅ {object_name} обойдено")
                    except Exception as e:
                        log.error(f"❌ Ошибка обхода {object_name}: {e}")
            elif detected['type'] == 'small':
                log.obstacle(f"ℹ️ {object_name} на расстоянии {detected['distance']:.1f} м")
            elif detected['type'] == 'environmental':
                log.warn(f"🌫️ Погодное явление: {object_name}")
        
        # Определение действия
        if not collision_detected:
            if detected and detected.get('type') in ['large']:
                action, confidence = 0, 0.8
            elif obstacle_count > 3 and flight_time > 20:
                action, confidence = 0, 0.6
            else:
                action, confidence = 1, 0.85
        
        # XAI ОБЪЯСНЕНИЕ (пока отключено до добавления LIME)
        if xai_enabled and not collision_detected and False:
            try:
                if flight_time % 5 == 0 or (detected and detected.get('type') in ['large', 'small']):
                    explanation = xai.explain_decision(
                        state=state,
                        action=action,
                        confidence=confidence,
                        method="grad_cam"
                    )
                    if explanation:
                        xai_explanations.append(explanation)
                        if explanation.get('is_anomaly', {}).get('is_anomaly', False):
                            log.warn(f"📊 Аномалия: {explanation['is_anomaly']['reasons']}")
                        log.info(f"📊 Решение: {explanation.get('action_name')} (уверенность: {explanation.get('confidence', 0):.2f})")
                        if explanation.get('runtime_ms', 0) > 0:
                            log.info(f"   ⏱️ XAI: {explanation['runtime_ms']:.1f} мс")
            except Exception as e:
                log.error(f"❌ XAI ошибка: {e}")
        
        # Координаты раз в 5 секунд
        if flight_time % 5 == 0:
            log.drone(f"📍 Координаты: X={pos['x']:.1f}, Y={pos['y']:.1f}, Z={pos['z']:.1f}")
            log.drone(f"🎯 До цели: {distance_to_target:.1f} м")
        
        time.sleep(1.0)

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
    log.ok("СТАТУС: ЦЕЛЬ ДОСТИГНУТА ✅")
else:
    log.ok("СТАТУС: ПОЛЁТ ЗАВЕРШЁН")

log.info(f"⏱️ Длительность: {flight_time} сек")
log.info(f"🚧 Всего препятствий: {obstacle_count}")

# 👉 СТАТИСТИКА ПО ОБЪЕКТАМ
print("\n" + "="*60)
print("📊 СТАТИСТИКА КОЛЛИЗИЙ ПО ОБЪЕКТАМ")
print("="*60)
total_collisions = sum(collision_stats.values())
for obj, count in collision_stats.items():
    if count > 0:
        pct = (count / total_collisions * 100) if total_collisions > 0 else 0
        log.info(f"   {obj.upper()}: {count} ({pct:.1f}%)")

log.info(f"📊 Всего коллизий: {total_collisions} из {MAX_COLLISION_COUNT}")

# 👉 ТОП-3 ОБЪЕКТА
if total_collisions > 0:
    sorted_objects = sorted(collision_stats.items(), key=lambda x: x[1], reverse=True)
    print("\n🏆 ТОП-3 ОБЪЕКТА-ПРЕПЯТСТВИЯ:")
    for i, (obj, count) in enumerate(sorted_objects[:3], 1):
        if count > 0:
            log.info(f"   {i}. {obj.upper()}: {count} раз")

# XAI ОТЧЁТ
if xai_enabled and xai_explanations:
    print("\n" + "="*60)
    print("📊 XAI ОТЧЁТ")
    print("="*60)
    try:
        xai_summary = xai.get_summary()
        log.info(f"Всего объяснений: {xai_summary.get('count', 0)}")
        log.info(f"Аномалий: {xai_summary.get('anomaly_count', 0)}")
        log.info(f"Доля: {xai_summary.get('anomaly_rate', 0.0):.1%}")
        xai_path = f"data/logs/xai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        xai.save_explanations(xai_path)
        log.info(f"XAI сохранены: {xai_path}")
    except Exception as e:
        log.error(f"❌ Ошибка XAI отчёта: {e}")
else:
    log.info("ℹ️ XAI объяснения не генерировались")

log.save()
adapter.disconnect()

print("\n✅ Мониторинг завершён!")