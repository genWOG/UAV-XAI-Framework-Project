"""
Отдельный процесс для чтения LiDAR через ROS 2.
Сохраняет данные в data/runtime/lidar_state.json для flight_monitor.py.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import json
import os
import tempfile

# Топик LiDAR для мира uav_test_world и модели x500_lidar_2d_0
TOPIC = '/world/uav_test_world/model/x500_lidar_2d_0/link/link/sensor/lidar_2d_v2/scan'
OUT_FILE = 'data/runtime/lidar_state.json'

class LidarBridge(Node):
    def __init__(self):
        super().__init__('lidar_bridge')
        self.sub = self.create_subscription(LaserScan, TOPIC, self.callback, 10)
        os.makedirs('data/runtime', exist_ok=True)
        self.get_logger().info(f'Подписан на LiDAR: {TOPIC}')

    def callback(self, msg):
        try:
            ranges = np.array(msg.ranges, dtype=np.float32)
            # Заменяем inf/nan на 100.0
            ranges = np.where(np.isfinite(ranges), ranges, 100.0)
            
            n = len(ranges)
            center = n // 2
            
            # Функция для получения минимума в секторе
            def get_sector_min(start_idx, end_idx):
                sector = ranges[start_idx:end_idx]
                valid = sector[sector < 90.0]
                return float(np.min(valid)) if len(valid) > 0 else 100.0

            # Рассчитываем сектора (1080 лучей, угол от -135 до +135)
            # 1 градус = 1080 / 270 = 4 луча
            front = get_sector_min(center - 60, center + 60) # +/- 15 градусов
            left = get_sector_min(center + 240, center + 360) # от 60 до 90 градусов
            right = get_sector_min(center - 360, center - 240) # от -90 до -60 градусов
            back = min(get_sector_min(0, 120), get_sector_min(n - 120, n)) # Края
            
            min_dist = float(np.min(ranges))
            density = float(np.sum(ranges < 10.0) / len(ranges))

            state = {
                'front': front,
                'left': left,
                'right': right,
                'back': back,
                'min': min_dist,
                'density': density,
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            }

            # Атомарная запись в файл
            fd, tmp_path = tempfile.mkstemp(dir='data/runtime', prefix='lidar_')
            with os.fdopen(fd, 'w') as f:
                json.dump(state, f)
            os.replace(tmp_path, OUT_FILE)
            
            # Логируем раз в 5 секунд, чтобы не засорять консоль
            self.get_logger().info(f'LiDAR: front={front:.1f}, min={min_dist:.1f}', throttle_duration_sec=5.0)
            
        except Exception as e:
            self.get_logger().error(f'Ошибка обработки скана: {e}')

def main():
    rclpy.init()
    node = LidarBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()