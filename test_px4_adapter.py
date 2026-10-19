"""
Адаптер для PX4 SITL через MAVSDK (mavsdk_grpc).

Внутри PX4 использует GPS-координаты (lat/lon/alt).
Мы пересчитываем их в локальные X/Y/Z (метры), чтобы остальной код
(flight_monitor.py, obstacle_detector.py, XAI) не пришлось переписывать.

Логика координат:
- X=0, Y=0 — точка, где дрон стоял в момент ПОДКЛЮЧЕНИЯ скрипта.
- Команда move_to(x, y) означает "лети в точку (x, y) относительно точки подключения".

Совместим по интерфейсу с airsim_adapter.AirSimAdapter.
"""

import asyncio
import math
import time
import threading

from mavsdk_grpc import System
from mavsdk_grpc.offboard import OffboardError, PositionNedYaw

# Константы для пересчёта градусов в метры
LAT_METERS_PER_DEG = 111320.0


class PX4Adapter:
    """
    Адаптер PX4 <-> остальной проект.
    Внутри держит постоянный asyncio-loop в отдельном потоке,
    чтобы остальной код мог вызывать синхронные методы.
    """

    def __init__(self, log_manager=None, use_simulation=False):
        self.log = log_manager
        self.connected = False
        self.armed = False

        # Начальная GPS-точка (origin полёта)
        self._origin_lat = None
        self._origin_lon = None

        # Последнее состояние
        self._last_state = self._empty_state()

        # Намерение движения (north, east, down, yaw)
        self._target_ned = None

        # Asyncio loop в отдельном потоке
        self._loop = None
        self._thread = None
        self._drone = None
        self._stop_flag = False

        # Ограничения
        self.max_altitude = 10.0
        self.min_altitude = 3.0

        self._log("🚁 Инициализация PX4Adapter...")

    # ------------------------------------------------------------------ #
    # Логирование
    # ------------------------------------------------------------------ #
    def _log(self, msg, level='info'):
        if self.log:
            if level == 'error':
                self.log.error(msg)
            elif level == 'warn':
                self.log.warn(msg)
            elif level == 'system':
                self.log.system(msg)
            elif level == 'debug':
                self.log.info(msg)
            else:
                self.log.info(msg)
        else:
            print(f"[{level.upper()}] {msg}")

    # ------------------------------------------------------------------ #
    # Пустое состояние
    # ------------------------------------------------------------------ #
    def _empty_state(self):
        return {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'velocity': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
            'sensors': {
                'obstacle_distance': 100.0,
                'obstacle_density': 0.0,
                'battery': 100.0,
                'collision': False,
                'object_name': None,
                'gps': {'lat': 0.0, 'lon': 0.0, 'alt': 0.0},
                'imu': {'accel': [0.0, 0.0, 0.0], 'gyro': [0.0, 0.0, 0.0]},
                'barometer': {'pressure': 1013.25, 'altitude': 0.0},
                'magnetometer': {'mag': [0.0, 0.0, 0.0]},
                'lidar_points': [],
                'distance_sensor_distance': 100.0,
                'distance_sensor_available': False,
            },
        }

    # ------------------------------------------------------------------ #
    # Подключение
    # ------------------------------------------------------------------ #
    def connect(self):
        if self._thread is not None:
            return self.connected

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Ждём подключения
        start = time.time()
        while not self.connected and time.time() - start < 15:
            time.sleep(0.2)

        if not self.connected:
            self._log("❌ Не удалось подключиться к PX4 за 15 секунд", 'error')
            return False

        # Ждём первую GPS-позицию
        self._log("⏳ Ожидание первой GPS-позиции для фиксации origin...", 'system')
        start = time.time()
        while self._origin_lat is None and time.time() - start < 10:
            time.sleep(0.2)

        if self._origin_lat is None:
            self._log("⚠️ Не удалось получить GPS-позицию", 'warn')
        else:
            self._log(
                f"✅ Origin зафиксирован: "
                f"{self._origin_lat:.6f}, {self._origin_lon:.6f}",
                'system'
            )
        self._log("✅ Подключение к PX4 установлено!", 'system')
        return True

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            self._log(f"❌ Ошибка в фоновом потоке: {e}", 'error')

    async def _main(self):
        self._drone = System()
        addr = "udpin://0.0.0.0:14540"
        self._log(f"🔌 Подключаемся к PX4: {addr}", 'system')
        await self._drone.connect(system_address=addr)

        async for state in self._drone.core.connection_state():
            if state.is_connected:
                self.connected = True
                break

        tasks = [
            asyncio.create_task(self._watch_position()),
            asyncio.create_task(self._watch_attitude()),
            asyncio.create_task(self._watch_battery()),
            asyncio.create_task(self._watch_imu()),
            asyncio.create_task(self._watch_baro()),
            asyncio.create_task(self._watch_mag()),
            asyncio.create_task(self._watch_velocity()),
        ]

        while not self._stop_flag:
            await asyncio.sleep(0.5)

        for t in tasks:
            t.cancel()

    # ------------------------------------------------------------------ #
    # Подписки на телеметрию
    # ------------------------------------------------------------------ #
    async def _watch_position(self):
        async for pos in self._drone.telemetry.position():
            lat = pos.latitude_deg
            lon = pos.longitude_deg
            alt = pos.relative_altitude_m

            if self._origin_lat is None:
                self._origin_lat = lat
                self._origin_lon = lon

            dx = (lat - self._origin_lat) * LAT_METERS_PER_DEG
            dy = (lon - self._origin_lon) * LAT_METERS_PER_DEG * math.cos(
                math.radians(self._origin_lat)
            )
            dz = -alt
            self._last_state['position'] = {'x': dx, 'y': dy, 'z': dz}
            self._last_state['sensors']['gps'] = {
                'lat': lat, 'lon': lon, 'alt': alt
            }

    async def _watch_attitude(self):
        async for att in self._drone.telemetry.attitude_euler():
            self._last_state['orientation'] = {
                'roll': att.roll_deg,
                'pitch': att.pitch_deg,
                'yaw': att.yaw_deg,
            }

    async def _watch_velocity(self):
        async for vel in self._drone.telemetry.velocity_ned():
            self._last_state['velocity'] = {
                'x': vel.north_m_s,
                'y': vel.east_m_s,
                'z': -vel.down_m_s,
            }

    async def _watch_battery(self):
        async for batt in self._drone.telemetry.battery():
            pct = batt.remaining_percent
            if pct <= 1.0:
                pct = pct * 100.0
            self._last_state['sensors']['battery'] = float(pct)

    async def _watch_imu(self):
        async for imu in self._drone.telemetry.imu():
            self._last_state['sensors']['imu'] = {
                'accel': [
                    imu.acceleration_frd.forward_m_s2,
                    imu.acceleration_frd.right_m_s2,
                    imu.acceleration_frd.down_m_s2,
                ],
                'gyro': [
                    imu.angular_velocity_frd.forward_rad_s,
                    imu.angular_velocity_frd.right_rad_s,
                    imu.angular_velocity_frd.down_rad_s,
                ],
            }

    async def _watch_baro(self):
        async for baro in self._drone.telemetry.barometer():
            self._last_state['sensors']['barometer'] = {
                'pressure': 1013.25,
                'altitude': float(baro.absolute_altitude_m),
            }

    async def _watch_mag(self):
        async for mag in self._drone.telemetry.magnetometer():
            self._last_state['sensors']['magnetometer'] = {
                'mag': [
                    mag.x_gauss,
                    mag.y_gauss,
                    mag.z_gauss,
                ],
            }

    # ------------------------------------------------------------------ #
    # Публичные методы (синхронные)
    # ------------------------------------------------------------------ #
    def _call(self, coro, timeout=60):
        if self._loop is None:
            return None
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=timeout)
        except Exception as e:
            self._log(f"❌ Ошибка вызова: {e}", 'error')
            return None

    def takeoff(self):
        self._log("🛫 Взлёт...")
        return self._call(self._takeoff_async())

    async def _takeoff_async(self):
        try:
            for attempt in range(3):
                try:
                    await self._drone.action.arm()
                    break
                except Exception as e:
                    self._log(f"⚠️ Arm попытка {attempt+1} не удалась: {e}", 'warn')
                    await asyncio.sleep(2)
            self.armed = True
            await self._drone.action.takeoff()
            await asyncio.sleep(5)
            return True
        except Exception as e:
            self._log(f"❌ Ошибка взлёта: {e}", 'error')
            return False

    def land(self):
        self._log("🛬 Посадка...")
        return self._call(self._land_async())

    async def _land_async(self):
        try:
            await self._drone.action.land()
            await asyncio.sleep(5)
            self.armed = False
            return True
        except Exception as e:
            self._log(f"❌ Ошибка посадки: {e}", 'error')
            return False

    def move_to(self, x, y, z, speed=5.0, force=False):
        """Движение к точке в локальных координатах (метры).
        x – север, y – восток, z – вниз (отрицательный = вверх).
        """
        alt = -z
        if alt > self.max_altitude:
            alt = self.max_altitude
        if alt < self.min_altitude:
            alt = self.min_altitude
        down = -alt
        self._target_ned = (x, y, down, 0.0)
        return self._call(self._move_async())

    async def _move_async(self):
        if self._target_ned is None:
            return False
        n, e, d, _ = self._target_ned
        try:
            target_yaw_deg = math.degrees(math.atan2(e, n))
            cur = self._last_state['position']
            cur_n = cur['x']
            cur_e = cur['y']
            cur_d = cur['z']

            self._log(
                f"🧭 Движение: цель=({n:.1f}, {e:.1f}, {d:.1f}), "
                f"старт=({cur_n:.1f}, {cur_e:.1f}, {cur_d:.1f}), "
                f"yaw={target_yaw_deg:.1f}°",
                'debug'
            )

            # Стартовая точка = текущая позиция
            await self._drone.offboard.set_position_ned(
                PositionNedYaw(cur_n, cur_e, cur_d, target_yaw_deg)
            )

            # Ретрай offboard.start()
            started = False
            for attempt in range(3):
                try:
                    await self._drone.offboard.start()
                    started = True
                    break
                except Exception as e:
                    self._log(
                        f"⚠️ offboard.start() попытка {attempt+1}: {e}",
                        'warn'
                    )
                    await asyncio.sleep(0.5)
            if not started:
                self._log("❌ Не удалось запустить offboard", 'error')
                return False

            start_time = time.time()
            max_wait = 10.0          # ← ФИКС: было 40.0
            min_send_interval = 0.1
            last_send = 0.0
            arrived = False

            while time.time() - start_time < max_wait:
                now = time.time()
                if now - last_send >= min_send_interval:
                    await self._drone.offboard.set_position_ned(
                        PositionNedYaw(n, e, d, target_yaw_deg)
                    )
                    last_send = now

                cur = self._last_state['position']
                dx = cur['x'] - n
                dy = cur['y'] - e
                dz = cur['z'] - d
                if abs(dx) < 2.0 and abs(dy) < 2.0 and abs(dz) < 1.5:
                    arrived = True
                    self._log(
                        f"🎯 Достигли цели: "
                        f"({cur['x']:.1f}, {cur['y']:.1f}, {cur['z']:.1f})",
                        'system'
                    )
                    break

                await asyncio.sleep(0.05)

            if not arrived:
                cur = self._last_state['position']
                self._log(
                    f"⚠️ Не долетели за {max_wait:.0f} сек. "
                    f"Текущая: ({cur['x']:.1f}, {cur['y']:.1f}, {cur['z']:.1f}), "
                    f"Цель: ({n:.1f}, {e:.1f}, {d:.1f})",
                    'warn'
                )

            try:
                await self._drone.offboard.stop()
            except OffboardError:
                pass
            return arrived

        except Exception as e:
            self._log(f"❌ Ошибка движения: {e}", 'error')
            return False

    def get_state(self):
        return self._last_state

    def get_gps_data(self):
        gps = self._last_state['sensors']['gps']
        return {
            'latitude': gps['lat'],
            'longitude': gps['lon'],
            'altitude': gps['alt'],
            'timestamp': time.time(),
            'is_real': True,
        }

    def disconnect(self):
        self._stop_flag = True
        self.connected = False
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._log("🔌 Отключение выполнено", 'system')