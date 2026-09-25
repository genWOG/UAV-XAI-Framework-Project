"""
Тест нового PX4Adapter.
Проверяет: подключение, взлёт, чтение телеметрии, посадку.
"""
import os
import sys
import time

# Добавляем пути
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.log_manager import LogManager
from modules.px4_adapter import PX4Adapter


def main():
    log = LogManager(log_dir="data/logs")

    print("=" * 60)
    print("🚁 ТЕСТ PX4ADAPTER")
    print("=" * 60)

    # 1. Создаём адаптер
    adapter = PX4Adapter(log)

    # 2. Подключаемся
    if not adapter.connect():
        print("❌ Не удалось подключиться")
        return

    # 3. Даём подпискам прогреться
    print("⏳ Ждём 3 секунды, чтобы телеметрия пришла...")
    time.sleep(3)

    # 4. Смотрим состояние ДО взлёта
    print("\n📊 СОСТОЯНИЕ ДО ВЗЛЁТА:")
    state = adapter.get_state()
    pos = state['position']
    gps = state['sensors']['gps']
    print(f"   Позиция: x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")
    print(f"   GPS:     lat={gps['lat']:.6f}, lon={gps['lon']:.6f}, alt={gps['alt']:.2f}")
    print(f"   Батарея: {state['sensors']['battery']:.1f}%")

    # 5. Взлёт
    print("\n🛫 Взлёт...")
    if not adapter.takeoff():
        print("❌ Не удалось взлететь")
        adapter.disconnect()
        return

    # 6. Ждём, пока наберёт высоту
    print("⏳ Ждём 5 секунд...")
    time.sleep(5)

    # 7. Смотрим состояние ПОСЛЕ взлёта
    print("\n📊 СОСТОЯНИЕ ПОСЛЕ ВЗЛЁТА:")
    state = adapter.get_state()
    pos = state['position']
    gps = state['sensors']['gps']
    vel = state['velocity']
    orient = state['orientation']
    print(f"   Позиция:   x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")
    print(f"   GPS:       lat={gps['lat']:.6f}, lon={gps['lon']:.6f}, alt={gps['alt']:.2f}")
    print(f"   Скорость:  vx={vel['x']:.2f}, vy={vel['y']:.2f}, vz={vel['z']:.2f}")
    print(f"   Ориентация: roll={orient['roll']:.1f}, pitch={orient['pitch']:.1f}, yaw={orient['yaw']:.1f}")
    print(f"   Батарея:   {state['sensors']['battery']:.1f}%")

    # 8. Двигаемся вперёд на 5 метров (x = север)
    print("\n🚁 Движение вперёд на 5 м...")
    adapter.move_to(5.0, 0.0, -5.0, speed=3.0)
    time.sleep(3)

    print("\n📊 СОСТОЯНИЕ ПОСЛЕ ДВИЖЕНИЯ:")
    state = adapter.get_state()
    pos = state['position']
    print(f"   Позиция:   x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}")

    # 9. Посадка
    print("\n🛬 Посадка...")
    adapter.land()
    time.sleep(5)

    # 10. Отключение
    adapter.disconnect()

    print("\n✅ ТЕСТ ЗАВЕРШЁН")


if __name__ == "__main__":
    main()