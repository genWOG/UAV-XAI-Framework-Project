"""
Анализ состояния датчиков и телеметрии
Поддерживает как объекты AirSim, так и словари
"""

import random
import time

class TelemetryParser:
    def __init__(self, log_manager):
        self.log = log_manager
        self.sensors = {
            'IMU': {'status': 'OK', 'last_value': None},
            'LiDAR': {'status': 'OK', 'last_value': None},
            'Camera': {'status': 'OK', 'last_value': None},
            'GPS': {'status': 'OK', 'last_value': None},
            'Barometer': {'status': 'OK', 'last_value': None}
        }
        self.errors = []

    def parse(self, state):
        """
        Анализирует телеметрию.
        Поддерживает как объекты AirSim, так и словари.
        """
        # Если state — словарь (из адаптера)
        if isinstance(state, dict):
            pos = state.get('position', {})
            vel = state.get('velocity', {})
            pos_x = pos.get('x', 0)
            pos_y = pos.get('y', 0)
            pos_z = pos.get('z', 0)
            vx = vel.get('vx', 0)
            vy = vel.get('vy', 0)
            vz = vel.get('vz', 0)
        else:
            # Если state — объект AirSim
            try:
                pos = state.kinematics_estimated.position
                vel = state.kinematics_estimated.linear_velocity
                pos_x = pos.x_val
                pos_y = pos.y_val
                pos_z = pos.z_val
                vx = vel.x_val
                vy = vel.y_val
                vz = vel.z_val
            except AttributeError:
                # Если структура неожиданная
                self.log.warn("Неизвестный формат данных, использую значения по умолчанию")
                pos_x, pos_y, pos_z = 0, 0, 0
                vx, vy, vz = 0, 0, 0
        
        # Проверка датчиков (симуляция)
        sensor_status = self._check_sensors()
        
        for sensor, data in sensor_status.items():
            if data['status'] != self.sensors[sensor]['status']:
                if data['status'] == 'ERROR':
                    self.log.error(f"Датчик {sensor} вышел из строя!")
                    self.errors.append({'sensor': sensor, 'time': time.time()})
                elif data['status'] == 'WARN':
                    self.log.warn(f"Датчик {sensor}: нестабильные показания")
                self.sensors[sensor]['status'] = data['status']
        
        return {
            'position': {'x': pos_x, 'y': pos_y, 'z': pos_z},
            'velocity': {'vx': vx, 'vy': vy, 'vz': vz},
            'sensors': sensor_status,
            'errors': self.errors[-5:] if self.errors else []
        }

    def _check_sensors(self):
        """Проверка состояния датчиков (симуляция)"""
        status = {}
        for sensor in self.sensors:
            if random.random() < 0.02:  # 2% ошибка
                status[sensor] = {'status': 'ERROR', 'last_value': None}
            elif random.random() < 0.05:  # 5% предупреждение
                status[sensor] = {'status': 'WARN', 'last_value': None}
            else:
                status[sensor] = {'status': 'OK', 'last_value': random.uniform(0, 100)}
        return status

    def get_sensor_summary(self):
        """Краткий отчёт по датчикам"""
        ok = sum(1 for s in self.sensors.values() if s['status'] == 'OK')
        total = len(self.sensors)
        errors = sum(1 for s in self.sensors.values() if s['status'] == 'ERROR')
        return f"Датчики: {ok}/{total} OK, ошибок: {errors}"