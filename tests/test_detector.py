"""
Тест обнаружения препятствий
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'xai'))

from modules.log_manager import LogManager
from modules.obstacle_detector import ObstacleDetector

print("🧪 ТЕСТ: Обнаружение препятствий")
print("="*40)

log = LogManager(log_dir="test_logs")
detector = ObstacleDetector(log)

# Тестовые данные
test_points = [
    (10, 5, -1, 5.0),
    (30, 20, -1, 5.0),
    (50, 40, -1, 5.0),
    (80, 70, -1, 5.0)
]

for x, y, z, speed in test_points:
    sensors = {'lidar': 30, 'camera': True}
    result = detector.detect(x, y, z, speed, sensors)
    if result:
        print(f"✅ Обнаружено: {result['object']} (тип: {result['type']}, дист: {result['distance']:.1f} м)")
    else:
        print(f"❌ Ничего не обнаружено в ({x}, {y})")

print("\n📊 Статистика:", detector.get_summary())
print("✅ Тест завершён!")