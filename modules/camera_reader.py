"""
Модуль чтения кадров с камеры дрона через ROS 2.

Работает в отдельном треде, не блокирует основной цикл.
Подписывается на топик sensor_msgs/Image, хранит последний кадр.
Также умеет сохранять кадры в файлы.
"""

import os
import time
import threading
from datetime import datetime

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


class CameraReader:
    """
    Читает кадры с ROS 2-топика камеры в фоне.
    """

    def __init__(self, log_manager=None,
                 topic='/world/ridge/model/x500_depth_0/link/camera_link/sensor/IMX214/image',
                 save_dir='results/frames'):
        self.log = log_manager
        self.topic = topic
        self.save_dir = save_dir
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.frames_count = 0
        self.running = False
        self._thread = None
        self._node = None

        if not ROS2_AVAILABLE:
            self._log("⚠️ ROS 2 (rclpy) не найден. Камера не будет работать.", 'warn')
            return

        os.makedirs(self.save_dir, exist_ok=True)

    def _log(self, msg, level='info'):
        if self.log:
            if level == 'error':
                self.log.error(msg)
            elif level == 'warn':
                self.log.warn(msg)
            elif level == 'system':
                self.log.system(msg)
            else:
                self.log.info(msg)
        else:
            print(f"[{level.upper()}] {msg}")

    def start(self):
        if not ROS2_AVAILABLE:
            return False
        if self._thread is not None and self._thread.is_alive():
            return True

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        start = time.time()
        while self.latest_frame is None and time.time() - start < 10:
            time.sleep(0.2)

        if self.latest_frame is None:
            self._log("⚠️ Первый кадр не получен за 10 сек", 'warn')
        else:
            self._log(f"✅ Камера работает: кадр {self.latest_frame.shape}", 'system')
        return True

    def _run(self):
        try:
            if not rclpy.ok():
                rclpy.init()
            self._node = _CameraNode(self)
            rclpy.spin(self._node)
        except Exception as e:
            self._log(f"❌ Ошибка в потоке камеры: {e}", 'error')
        finally:
            try:
                if self._node is not None:
                    self._node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def get_latest_frame(self):
        return self.latest_frame

    def get_latest_timestamp(self):
        return self.latest_timestamp

    def save_frame(self, filename=None):
        if self.latest_frame is None:
            self._log("⚠️ save_frame: кадра нет в памяти", 'warn')
            return None

        if filename is None:
            filename = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"

        filepath = os.path.join(self.save_dir, filename)

        try:
            from PIL import Image
            img = Image.fromarray(self.latest_frame)
            img.save(filepath)
            return filepath
        except ImportError:
            npy_path = filepath.replace('.png', '.npy')
            np.save(npy_path, self.latest_frame)
            return npy_path
        except Exception as e:
            self._log(f"❌ Ошибка сохранения кадра: {e}", 'error')
            return None

    def stop(self):
        self.running = False
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass


# ============================================================
# ROS 2 Node
# ============================================================

if ROS2_AVAILABLE:
    class _CameraNode(Node):
        def __init__(self, reader: CameraReader):
            super().__init__('uav_camera_reader')
            self.reader = reader
            self.sub = self.create_subscription(
                Image, reader.topic, self.callback, 10
            )
            reader._log(f"📷 Подписан на топик: {reader.topic}", 'system')

        def callback(self, msg):
            try:
                height = msg.height
                width = msg.width
                encoding = msg.encoding

                if encoding == 'rgb8' or encoding == 'bgr8':
                    channels = 3
                    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        (height, width, channels)
                    )
                    if encoding == 'bgr8':
                        arr = arr[:, :, ::-1]
                elif encoding == 'mono8':
                    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        (height, width)
                    )
                elif encoding in ('16UC1', '32FC1'):
                    dtype = np.uint16 if encoding == '16UC1' else np.float32
                    arr = np.frombuffer(msg.data, dtype=dtype).reshape(
                        (height, width)
                    )
                else:
                    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                        (height, width, -1)
                    )

                self.reader.latest_frame = arr.copy()
                self.reader.latest_timestamp = time.time()
                self.reader.frames_count += 1

                if self.reader.frames_count % 10 == 1:
                    self.reader._log(
                        f"📷 Кадров получено: {self.reader.frames_count}, "
                        f"форма {arr.shape}, encoding={encoding}",
                        'info'
                    )
            except Exception as e:
                self.reader._log(f"❌ Ошибка обработки кадра: {e}", 'error')