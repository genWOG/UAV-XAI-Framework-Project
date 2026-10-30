import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import json
import os
import time
from ultralytics import YOLO

class VisionBridge(Node):
    def __init__(self):
        super().__init__('vision_bridge')
        
        # Загружаем нашу YOLO
        self.model = YOLO('runs/detect/uav_obstacles_v2/weights/best.pt')
        
        self.topic = '/world/uav_test_world/model/x500_depth_0/link/camera_link/sensor/IMX214/image'
        self.sub = self.create_subscription(Image, self.topic, self.callback, 10)
        
        self.output_file = 'data/runtime/vision_state.json'
        os.makedirs('data/runtime', exist_ok=True)
        
        self.last_infer_time = 0
        self.get_logger().info(f'👁️ Vision Bridge запущен. Подписан на {self.topic}')

    def callback(self, msg):
        now = time.time()
        # Делаем инференс раз в 1 секунду, чтобы не перегружать CPU
        if now - self.last_infer_time < 1.0:
            return
        self.last_infer_time = now

        try:
            height = msg.height
            width = msg.width
            
            if msg.encoding == "rgb8":
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "bgr8":
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
            else:
                self.get_logger().warn(f"Неизвестный формат: {msg.encoding}")
                return

            # --- ОТЛАДКА: Сохраняем кадр и прогоняем YOLO по нему ---
            cv2.imwrite('results/debug_array_frame.jpg', frame)
            # Прогоняем YOLO как по файлу
            file_results = self.model('results/debug_array_frame.jpg', conf=0.15, device='cpu', verbose=False)[0]
            
            # Прогоняем YOLO как по массиву
            array_results = self.model(frame, conf=0.15, device='cpu', verbose=False)[0]
            
            self.get_logger().info(f"YOLO(файл): {len(file_results.boxes)} | YOLO(массив): {len(array_results.boxes)}")
            
            # Берем результаты из массива
            results = array_results
            
            min_dist = 100.0
            detected_class = "none"
            
            debug_img = frame.copy()
            
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                bbox_area = (x2 - x1) * (y2 - y1)
                frame_area = width * height
                area_ratio = bbox_area / frame_area
                
                if area_ratio > 0.15: dist = 2.0
                elif area_ratio > 0.05: dist = 5.0
                elif area_ratio > 0.01: dist = 10.0
                else: dist = 20.0
                
                if dist < min_dist:
                    min_dist = dist
                    detected_class = self.model.names[cls]
                
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(debug_img, f"{self.model.names[cls]} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite('results/debug_frame_boxes.jpg', debug_img)

            state = {
                "front": min_dist,
                "detected_class": detected_class,
                "timestamp": now
            }
            
            with open(self.output_file + '.tmp', 'w') as f:
                json.dump(state, f)
            os.replace(self.output_file + '.tmp', self.output_file)
            
            if min_dist < 15.0:
                self.get_logger().info(f'⚠️ Вижу препятствие: {detected_class} (примерно {min_dist} м)')

        except Exception as e:
            self.get_logger().error(f'Ошибка Vision: {e}')

def main():
    rclpy.init()
    node = VisionBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()