"""
Адаптер для подключения к AirSim / Cosys-AirSim.

Версия 4.7 — ограничение высоты + GPS
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
        self.airsim_module = None
        self.lidar_available = True
        self.distance_sensor_available = False
        self.max_altitude = 10  # 👈 МАКСИМАЛЬНАЯ ВЫСОТА (10 м)
        self.min_altitude = 3    # 👈 МИНИМАЛЬНАЯ ВЫСОТА (3 м)
        self._import_airsim()

    def _import_airsim(self):
        """Импортирует AirSim и устанавливает соединение."""
        try:
            import cosysairsim as airsim
            self.airsim_version = 'cosys'
            self.airsim_module = airsim
            self.log.ok("✅ Используется Cosys-AirSim (UE5)")
        except ImportError:
            pass

        if self.airsim_module is None:
            try:
                import airsim
                self.airsim_version = 'original'
                self.airsim_module = airsim
                self.log.ok("✅ Используется оригинальный AirSim (Microsoft)")
            except ImportError:
                self.log.error("❌ AirSim не установлен!")
                return

        try:
            self.log.system("🔌 Подключение к AirSim...")
            self.client = self.airsim_module.MultirotorClient()
            self.client.confirmConnection()
            self.connected = True
            self.log.ok("✅ Подключение к AirSim установлено!")
            self._check_lidar()
            self._check_distance_sensor()
        except Exception as e:
            self.log.error(f"❌ Ошибка подключения: {e}")
            self.connected = False

    def _check_lidar(self):
        try:
            lidar_data = self.client.getLidarData()
            if lidar_data and hasattr(lidar_data, 'point_cloud'):
                self.lidar_available = True
                self.log.ok("✅ LiDAR доступен")
            else:
                self.lidar_available = False
                self.log.warn("⚠️ LiDAR не доступен")
        except Exception:
            self.lidar_available = False

    def _check_distance_sensor(self):
        try:
            distance_data = self.client.getDistanceSensorData()
            if distance_data and hasattr(distance_data, 'distance'):
                dist = float(distance_data.distance)
                if dist > 0 and dist < 1000:
                    self.distance_sensor_available = True
                    self.log.ok(f"✅ DistanceSensor доступен (расстояние: {dist:.1f} м)")
                    return
            self.distance_sensor_available = False
            self.log.warn("⚠️ DistanceSensor не доступен")
        except Exception as e:
            self.distance_sensor_available = False
            self.log.warn(f"⚠️ DistanceSensor ошибка: {e}")

    def takeoff(self) -> bool:
        if not self.connected:
            self.log.error("❌ Нет подключения к AirSim")
            return False

        try:
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.takeoffAsync()
            time.sleep(2)

            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position

            if abs(pos.z_val) < 2.0:
                self.log.warn("⚠️ Дрон не взлетел! Принудительный подъём...")
                self.client.moveToPositionAsync(pos.x_val, pos.y_val, -self.min_altitude, 5.0, timeout_sec=3.0).join()
                time.sleep(1)

            self.log.ok("✅ Взлёт выполнен!")
            return True

        except Exception as e:
            self.log.error(f"❌ Ошибка взлёта: {e}")
            return False

    def land(self) -> bool:
        if not self.connected:
            return False

        try:
            self.client.landAsync()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            self.log.ok("✅ Посадка выполнена!")
            return True
        except Exception as e:
            self.log.error(f"❌ Ошибка посадки: {e}")
            return False

    def move_to(self, x: float, y: float, z: float, speed: float, force: bool = False) -> bool:
        """
        Движение к точке с ограничением высоты.
        """
        if not self.connected:
            self.log.error("❌ Нет подключения к AirSim")
            return False

        # 👈 ОГРАНИЧЕНИЕ ВЫСОТЫ
        # Если цель ниже -max_altitude — опускаем до -max_altitude
        if z < -self.max_altitude:
            z = -self.max_altitude
            self.log.info(f"📏 Высота ограничена: {self.max_altitude} м")

        # Если цель выше -min_altitude (ближе к земле) — поднимаем
        if z > -self.min_altitude:
            z = -self.min_altitude

        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position

            dx = x - pos.x_val
            dy = y - pos.y_val
            dz = z - pos.z_val
            dist = (dx*dx + dy*dy + dz*dz) ** 0.5

            if dist > 0.5:
                airsim = self.airsim_module

                self.client.moveToPositionAsync(
                    x, y, z,
                    speed,
                    timeout_sec=3.0,
                    drivetrain=airsim.DrivetrainType.ForwardOnly,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0)
                ).join()

                self.log.ok(f"✅ Перемещение к ({x:.1f}, {y:.1f}, {z:.1f}) завершено")
                return True
            else:
                self.log.ok("✅ Уже на месте!")
                return True

        except Exception as e:
            self.log.error(f"❌ Ошибка перемещения: {e}")
            return False

    def get_gps_data(self):
        """Возвращает GPS данные."""
        try:
            gps = self.client.getGpsData()
            if gps and hasattr(gps, 'gnss'):
                return {
                    'latitude': gps.gnss.geo_point.latitude,
                    'longitude': gps.gnss.geo_point.longitude,
                    'altitude': gps.gnss.geo_point.altitude,
                    'timestamp': time.time()
                }
        except Exception:
            pass
        return None

    def get_distance_to_obstacle(self) -> float:
        if not self.distance_sensor_available:
            return 100.0

        try:
            data = self.client.getDistanceSensorData()
            if data and hasattr(data, 'distance'):
                return float(data.distance)
        except Exception:
            pass
        return 100.0

    def get_state(self):
        if not self.connected:
            return None

        try:
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            orient = state.kinematics_estimated.orientation

            roll, pitch, yaw = self._quaternion_to_euler(
                orient.x_val, orient.y_val, orient.z_val, orient.w_val
            )

            sensor_data = self._get_sensor_data()

            return {
                'position': {'x': pos.x_val, 'y': pos.y_val, 'z': pos.z_val},
                'velocity': {'x': vel.x_val, 'y': vel.y_val, 'z': vel.z_val},
                'orientation': {'roll': roll, 'pitch': pitch, 'yaw': yaw},
                'sensors': sensor_data
            }
        except Exception as e:
            self.log.warn(f"⚠️ Ошибка получения данных: {e}")
            self.connected = False
            return None

    def _get_sensor_data(self):
        sensor_data = {
            'obstacle_distance': 100.0,
            'obstacle_density': 0.0,
            'battery': 100.0,
            'collision': False,
            'object_name': None,
            'gps': {'lat': 0.0, 'lon': 0.0, 'alt': 0.0},
            'imu': {'accel': [0, 0, 0], 'gyro': [0, 0, 0]},
            'barometer': {'pressure': 1013.25, 'altitude': 0.0},
            'magnetometer': {'mag': [0, 0, 0]},
            'camera': None,
            'lidar_points': [],
            'distance_sensor_distance': 100.0,
            'distance_sensor_available': self.distance_sensor_available,
        }

        # LiDAR
        if self.lidar_available:
            try:
                lidar_data = self.client.getLidarData()
                if lidar_data and hasattr(lidar_data, 'point_cloud'):
                    points = np.array(lidar_data.point_cloud).reshape(-1, 3)
                    if len(points) > 0:
                        sensor_data['lidar_points'] = points.tolist()
                        distances = np.linalg.norm(points, axis=1)
                        min_dist = float(np.min(distances))
                        if min_dist < sensor_data['obstacle_distance']:
                            sensor_data['obstacle_distance'] = min_dist
                            sensor_data['obstacle_density'] = float(len(distances[distances < 10]) / len(distances))
                    else:
                        self.lidar_available = False
            except Exception:
                self.lidar_available = False

        # DistanceSensor
        try:
            distance_data = self.client.getDistanceSensorData()
            if distance_data and hasattr(distance_data, 'distance'):
                dist = float(distance_data.distance)
                sensor_data['distance_sensor_distance'] = dist
                sensor_data['distance_sensor_available'] = True
                if dist < sensor_data['obstacle_distance']:
                    sensor_data['obstacle_distance'] = dist
                if dist < 30:
                    self.log.info(f"📡 DistanceSensor: {dist:.1f} м")
        except Exception:
            pass

        # 👉 GPS (УЛУЧШЕННЫЙ)
        try:
            gps = self.client.getGpsData()
            if gps and hasattr(gps, 'gnss'):
                sensor_data['gps'] = {
                    'lat': gps.gnss.geo_point.latitude,
                    'lon': gps.gnss.geo_point.longitude,
                    'alt': gps.gnss.geo_point.altitude
                }
                # 👈 ЛОГИРУЕМ GPS, ЕСЛИ ОН ЕСТЬ
                if abs(gps.gnss.geo_point.latitude) > 0.01:
                    self.log.info(f"🌍 GPS: {gps.gnss.geo_point.latitude:.6f}, {gps.gnss.geo_point.longitude:.6f}")
        except Exception as e:
            pass

        # Столкновения
        try:
            collision_info = self.client.simGetCollisionInfo()
            if collision_info and hasattr(collision_info, 'has_collided'):
                if collision_info.has_collided:
                    sensor_data['collision'] = True
                    if hasattr(collision_info, 'object_name'):
                        sensor_data['object_name'] = collision_info.object_name
                    if hasattr(collision_info, 'impact_point'):
                        impact = collision_info.impact_point
                        pos = self.client.getMultirotorState().kinematics_estimated.position
                        dx = impact.x_val - pos.x_val
                        dy = impact.y_val - pos.y_val
                        dz = impact.z_val - pos.z_val
                        dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                        sensor_data['obstacle_distance'] = float(dist)
        except:
            pass

        # IMU
        try:
            imu = self.client.getImuData()
            if imu:
                sensor_data['imu'] = {
                    'accel': [
                        imu.linear_acceleration.x_val,
                        imu.linear_acceleration.y_val,
                        imu.linear_acceleration.z_val
                    ],
                    'gyro': [
                        imu.angular_velocity.x_val,
                        imu.angular_velocity.y_val,
                        imu.angular_velocity.z_val
                    ]
                }
        except:
            pass

        # Barometer
        try:
            baro = self.client.getBarometerData()
            if baro:
                sensor_data['barometer'] = {
                    'pressure': baro.pressure,
                    'altitude': baro.altitude
                }
        except:
            pass

        # Battery
        try:
            battery = self.client.getBatteryState()
            if battery and hasattr(battery, 'percentage'):
                sensor_data['battery'] = float(battery.percentage)
        except:
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
                'object_name': None,
                'gps': {'lat': 0.0, 'lon': 0.0, 'alt': 0.0},
                'imu': {'accel': [0, 0, 0], 'gyro': [0, 0, 0]},
                'barometer': {'pressure': 1013.25, 'altitude': 0.0},
                'magnetometer': {'mag': [0, 0, 0]},
                'camera': None,
                'lidar_points': [],
                'distance_sensor_distance': 100.0,
                'distance_sensor_available': False,
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

    def get_camera_image(self):
        try:
            import airsim
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            if responses:
                img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img = img.reshape(responses[0].height, responses[0].width, 3)
                return img
        except:
            pass
        return None

    def disconnect(self):
        self.connected = False
        self.client = None
        self.log.system("🔌 Отключение выполнено")