"""
Тест GPS в AirSim
"""

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))
from airsim_adapter import AirSimAdapter
from log_manager import LogManager

log = LogManager()
adapter = AirSimAdapter(log, use_simulation=False)

if not adapter.connected:
    print("❌ Нет подключения к симулятору!")
    sys.exit(1)

print("\n" + "=" * 60)
print("🔍 ПРОВЕРКА GPS")
print("=" * 60)

# Пробуем получить GPS
for i in range(5):
    gps = adapter.get_gps_data()
    if gps:
        print(f"✅ GPS получен: {gps}")
        print(f"   Широта: {gps['latitude']:.6f}")
        print(f"   Долгота: {gps['longitude']:.6f}")
        print(f"   Высота: {gps['altitude']:.1f} м")
        break
    else:
        print(f"⚠️ Попытка {i+1}: GPS не получен...")
        time.sleep(1)

# Проверяем сенсоры
print("\n📡 ПРОВЕРКА СЕНСОРОВ")
state = adapter.get_state()
if state:
    sensors = state.get('sensors', {})
    gps_sensor = sensors.get('gps', {})
    print(f"GPS в сенсорах: {gps_sensor}")

adapter.disconnect()