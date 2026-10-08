"""
Отдельный скрипт для записи кадров с камеры дрона.
Работает независимо от flight_monitor.py.

Версия 2:
- кадры уменьшаются до 640x360 и сохраняются в JPEG (вместо PNG 1080p):
  кодирование намного дешевле для CPU и кадры занимают ~50 КБ вместо мегабайт
- интервал 0.5 с (камера даёт ~2 Гц)
- рядом пишется индекс frames_index.csv (t, filename), где t — время Unix,
  то же время, что и в телеметрии (incident_logger.py), по нему кадры
  связываются с телеметрией и выходами модели

ВАЖНО: этот скрипт использует rclpy (ROS 2).
flight_monitor.py использует mavsdk_grpc.
Они НЕ должны работать в одном процессе — иначе mavsdk_server падает.
"""

import os
import sys
import time
from datetime import datetime

import numpy as np

# ROS 2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


# Параметры
TOPIC = "/world/uav_test_world/model/x500_depth_0/link/camera_link/sensor/IMX214/image"
SAVE_DIR = "results/frames"
SAVE_INTERVAL = 0.5  # секунд между сохранениями
OUT_SIZE = (640, 360)  # (ширина, высота) сохраняемого кадра
JPEG_QUALITY = 90
INDEX_PATH = os.path.join(SAVE_DIR, "frames_index.csv")


class CameraRecorder(Node):
    def __init__(self):
        super().__init__("camera_recorder")
        self.latest_frame = None
        self.last_save = 0.0
        self.frames_count = 0
        self.saved_count = 0

        self.sub = self.create_subscription(
            Image, TOPIC, self.callback, 10
        )
        self.get_logger().info(f"📷 Подписан на {TOPIC}")

        os.makedirs(SAVE_DIR, exist_ok=True)
        new_index = not os.path.exists(INDEX_PATH)
        self.index_file = open(INDEX_PATH, "a", encoding="utf-8")
        if new_index:
            self.index_file.write("t,filename\n")
            self.index_file.flush()

    def callback(self, msg):
        try:
            self.frames_count += 1

            # Сохраняем раз в SAVE_INTERVAL секунд
            now = time.time()
            if now - self.last_save < SAVE_INTERVAL:
                return
            self.last_save = now

            # Конвертация
            height = msg.height
            width = msg.width
            encoding = msg.encoding

            if encoding == "rgb8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (height, width, 3)
                )
            elif encoding == "bgr8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    (height, width, 3)
                )[:, :, ::-1]
            else:
                self.get_logger().warn(f"Неизвестный encoding: {encoding}")
                return

            # Сохраняем (время берём в момент получения кадра)
            t_frame = now
            filename = (
                f"frame_{datetime.fromtimestamp(t_frame).strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            )
            filepath = os.path.join(SAVE_DIR, filename)

            from PIL import Image as PILImage
            img = PILImage.fromarray(arr).resize(OUT_SIZE, PILImage.BILINEAR)
            img.save(filepath, format="JPEG", quality=JPEG_QUALITY)

            self.index_file.write(f"{t_frame:.6f},{filename}\n")
            self.index_file.flush()

            self.saved_count += 1
            if self.saved_count % 20 == 1:
                self.get_logger().info(
                    f"📷 Кадр #{self.saved_count} сохранён: {filename} "
                    f"(всего получено: {self.frames_count})"
                )
        except Exception as e:
            self.get_logger().error(f"Ошибка: {e}")


def main():
    rclpy.init()
    node = CameraRecorder()
    print(f"\n📷 Camera Recorder запущен")
    print(f"   Топик: {TOPIC}")
    print(f"   Папка: {SAVE_DIR}")
    print(f"   Интервал: {SAVE_INTERVAL} сек\n")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"\n📷 Остановлено. Сохранено кадров: {node.saved_count}")
    finally:
        node.index_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()