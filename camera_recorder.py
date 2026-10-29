import os
import time
import json
import numpy as np
from datetime import datetime
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from modules.frame_queue import get_pending_request, remove_request
    QUEUE_AVAILABLE = True
except ImportError:
    QUEUE_AVAILABLE = False
    print("⚠️ Модуль frame_queue не найден.")

class CameraRecorder(Node):
    def __init__(self):
        super().__init__('camera_recorder')
        
        self.topic = '/world/uav_test_world/model/x500_depth_0/link/camera_link/sensor/IMX214/image'
        self.sub = self.create_subscription(Image, self.topic, self.callback, 10)
        self.last_save_time = 0 
        
        self.save_dir = 'results/frames'
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.frames_count = 0
        self.saved_count = 0
        
        self.get_logger().info(f'📷 Подписан на {self.topic}')
        print(f"\n📷 Camera Recorder запущен (режим: только инциденты)")

    def callback(self, msg):
        try:
            self.frames_count += 1
            
            # Проверяем, есть ли запрос на кадр
            req = None
            if QUEUE_AVAILABLE:
                req = get_pending_request()
            
            # Если запроса нет — сразу выходим (ничего не сохраняем)
            if req is None:
                return
                
            # --- ДОБАВИТЬ ЗАЩИТУ ОТ СПАМА ---
            now = time.time()
            if now - self.last_save_time < 2.0: # Не чаще 1 раза в 2 секунды
                return
            self.last_save_time = now
            # --------------------------------                
            
            # Готовим изображение
            height = msg.height
            width = msg.width
            encoding = msg.encoding
            
            if encoding == "rgb8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
            elif encoding == "bgr8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))[:, :, ::-1]
            else:
                return
                
            # Формируем имя с меткой события
            event_kind = req['event_kind']
            filename = f"frame_{event_kind}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            
            # Сохраняем
            img = PILImage.fromarray(arr)
            img.save(filepath)
            self.saved_count += 1
            self.get_logger().info(f"📸 ИНЦИДЕНТ: {event_kind} | Сохранён: {filename}")
            
            # Удаляем запрос из очереди
            if req is not None and QUEUE_AVAILABLE:
                remove_request(req["filename"])
                
        except Exception as e:
            self.get_logger().error(f"Ошибка: {e}")

def main():
    rclpy.init()
    node = CameraRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()