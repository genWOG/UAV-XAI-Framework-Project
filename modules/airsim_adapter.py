"""
Адаптер для подключения к AirSim / Cosys-AirSim.
"""

import time
import numpy as np


class AirSimAdapter:
    def __init__(self, log_manager, use_simulation: bool = False):
        self.log = log_manager
        self.use_simulation = use_simulation
        self.client = None
        self.connected = False
        self.airsim_version = None
        
        self._import_airsim()
    
    def _import_airsim(self):
        try:
            import cosysairsim as airsim
            self.airsim_version = 'cosys'
            self.log.ok("✅ Используется Cosys-AirSim (UE5)")
        except ImportError:
            pass
        
        if self.airsim_version is None:
            try:
                import airsim
                self.airsim_version = 'original'
                self.log.ok("✅ Используется оригинальный AirSim (Microsoft)")
            except ImportError:
                self.log.error("❌ AirSim не установлен")
                return
        
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            self.connected = True
            self.log.ok("✅ Подключение к AirSim установлено!")
        except Exception as e:
            self.log.error(f"❌ Ошибка подключения: {e}")
            self.connected = False
    
    def takeoff(self) -> bool:
        if not self.connected:
            self.log.error("❌ Нет подключения к AirSim")
            return False
        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync().join()
            self.log.ok("✅ Взлёт выполнен!")
            return True
        except Exception as e:
            self.log.error(f"❌ Ошибка взлёта: {e}")
            return False
    
    def land(self) -> bool:
        if not self.connected:
            return False
        try:
            self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            self.log.ok("✅ Посадка выполнена!")
            return True
        except Exception as e:
            self.log.error(f"❌ Ошибка посадки: {e}")
            return False
    
    def move_to(self, x: float, y: float, z: float, speed: float) -> bool:
        if not self.connected:
            return False
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            
            dx = x - pos.x_val
            dy = y - pos.y_val
            dz = z - pos.z_val
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5
            
            if dist > 0.5:
                vx = (dx / dist) * speed
                vy = (dy / dist) * speed
                vz = (dz / dist) * speed
                self.client.moveByVelocityAsync(vx, vy, vz, 0.5).join()
                self.log.ok(f"✅ Перемещение к ({x:.1f}, {y:.1f}, {z:.1f})")
                return True
            else:
                self.log.ok("✅ Уже на месте!")
                return True
        except Exception as e:
            self.log.error(f"❌ Ошибка перемещения: {e}")
            return False
    
    def get_state(self):
        if not self.connected:
            return self._empty_state()
        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            orient = state.kinematics_estimated.orientation
            
            roll, pitch, yaw = self._quaternion_to_euler(
                orient.x_val, orient.y_val, orient.z_val, orient.w_val
            )
            
            # 👉 ПОЛУЧАЕМ ДАННЫЕ О ПРЕПЯТСТВИЯХ
            sensor_data = self._get_sensor_data()
            
            return {
                'position': {'x': pos.x_val, 'y': pos.y_val, 'z': pos.z_val},
                'velocity': {'x': vel.x_val, 'y': vel.y_val, 'z': vel.z_val},
                'orientation': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
                'sensors': sensor_data
            }
        except Exception as e:
            self.log.warn(f"⚠️ Ошибка получения данных: {e}")
            return self._empty_state()
    
    def _get_sensor_data(self):
        """
        Получение данных с сенсоров.
        Для Cosys-AirSim используем несколько методов.
        """
        sensor_data = {
            'obstacle_distance': 100.0,
            'obstacle_density': 0.0,
            'battery': 100.0,
            'collision': False,
            'object_name': None
        }
        
        # ============================================
        # МЕТОД 1: Получение данных о столкновениях
        # ============================================
        try:
            collision_info = self.client.simGetCollisionInfo()
            if collision_info and hasattr(collision_info, 'has_collided'):
                if collision_info.has_collided:
                    sensor_data['collision'] = True
                    if hasattr(collision_info, 'object_name'):
                        sensor_data['object_name'] = collision_info.object_name
                    if hasattr(collision_info, 'impact_point'):
                        # Вычисляем расстояние до точки столкновения
                        impact = collision_info.impact_point
                        pos = self.client.getMultirotorState().kinematics_estimated.position
                        dx = impact.x_val - pos.x_val
                        dy = impact.y_val - pos.y_val
                        dz = impact.z_val - pos.z_val
                        dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                        sensor_data['obstacle_distance'] = float(dist)
                    self.log.warn(f"⚠️ Обнаружено столкновение с {sensor_data['object_name']}")
        except Exception as e:
            pass
        
        # ============================================
        # МЕТОД 2: Получение данных с LiDAR
        # ============================================
        try:
            lidar_data = self.client.getLidarData()
            if lidar_data and hasattr(lidar_data, 'point_cloud'):
                points = np.array(lidar_data.point_cloud).reshape(-1, 3)
                if len(points) > 0:
                    distances = np.linalg.norm(points, axis=1)
                    min_dist = float(np.min(distances))
                    # Обновляем расстояние только если оно меньше текущего
                    if min_dist < sensor_data['obstacle_distance']:
                        sensor_data['obstacle_distance'] = min_dist
                    sensor_data['obstacle_density'] = float(len(distances[distances < 10]) / len(distances))
        except Exception as e:
            pass
        
        # ============================================
        # МЕТОД 3: DistanceSensor (если доступен)
        # ============================================
        try:
            distance_data = self.client.getDistanceSensorData()
            if distance_data and hasattr(distance_data, 'distance'):
                dist = float(distance_data.distance)
                if dist < sensor_data['obstacle_distance']:
                    sensor_data['obstacle_distance'] = dist
        except Exception as e:
            pass
        
        # ============================================
        # МЕТОД 4: Получение данных о батарее
        # ============================================
        try:
            battery = self.client.getBatteryState()
            if battery and hasattr(battery, 'percentage'):
                sensor_data['battery'] = float(battery.percentage)
        except Exception:
            pass
        
        return sensor_data
    
    def _empty_state(self):
        return {
            'position': {'x': 0, 'y': 0, 'z': 0},
            'velocity': {'x': 0, 'y': 0, 'z': 0},
            'orientation': {'roll': 0, 'pitch': 0, 'yaw': 0},
            'sensors': {
                'obstacle_distance': 100.0,
                'obstacle_density': 0.0,
                'battery': 100.0,
                'collision': False,
                'object_name': None
            }
        }
    
    @staticmethod
    def _quaternion_to_euler(x, y, z, w):
        import math
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw
    
    def disconnect(self):
        self.connected = False
        self.client = None
        self.log.system("🔌 Отключение выполнено")