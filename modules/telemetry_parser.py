"""
Модуль парсинга телеметрии для БПЛА.
"""

import time
import math

class TelemetryParser:
    """
    Парсер телеметрии для анализа данных с сенсоров.
    """
    
    def __init__(self, log_manager=None):
        self.log = log_manager
        self.last_telemetry = None
        self.telemetry_history = []
        self.sensor_errors = []
    
    def parse(self, state):
        """
        Парсинг состояния БПЛА в структурированную телеметрию.
        """
        pos = state.get('position', {})
        vel = state.get('velocity', {})
        orient = state.get('orientation', {})
        sensors = state.get('sensors', {})
        
        # 👉 РЕАЛЬНАЯ ВЫСОТА (из позиции)
        real_altitude = abs(pos.get('z', 0))  # Модуль Z = высота в метрах
        
        # 👉 Данные с барометра (если есть)
        baro = sensors.get('barometer', {})
        baro_altitude = baro.get('altitude', 0)
        
        # 👉 GPS (если есть)
        gps = sensors.get('gps', {})
        
        telemetry = {
            'timestamp': time.time(),
            'position': {
                'x': pos.get('x', 0),
                'y': pos.get('y', 0),
                'z': pos.get('z', 0),
                'altitude': real_altitude,  # 👈 РЕАЛЬНАЯ ВЫСОТА
            },
            'velocity': {
                'x': vel.get('x', 0),
                'y': vel.get('y', 0),
                'z': vel.get('z', 0),
                'speed': math.sqrt(vel.get('x', 0)**2 + vel.get('y', 0)**2 + vel.get('z', 0)**2)
            },
            'orientation': {
                'roll': orient.get('roll', 0),
                'pitch': orient.get('pitch', 0),
                'yaw': orient.get('yaw', 0)
            },
            'sensors': {
                'obstacle_distance': sensors.get('obstacle_distance', 100.0),
                'obstacle_density': sensors.get('obstacle_density', 0.0),
                'battery': sensors.get('battery', 100.0),
                'collision': sensors.get('collision', False),
                'object_name': sensors.get('object_name', None),
                'gps': gps,
                'barometer': {
                    'pressure': baro.get('pressure', 1013.25),
                    'altitude': baro_altitude,
                    'real_altitude': real_altitude  # 👈 ДЛЯ СРАВНЕНИЯ
                },
                'imu': sensors.get('imu', {'accel': [0,0,0], 'gyro': [0,0,0]}),
                'magnetometer': sensors.get('magnetometer', {'mag': [0,0,0]}),
                'lidar_points_count': len(sensors.get('lidar_points', []))
            },
            'errors': self._check_errors(state, real_altitude)
        }
        
        self.last_telemetry = telemetry
        self.telemetry_history.append(telemetry)
        
        # Ограничиваем историю
        if len(self.telemetry_history) > 1000:
            self.telemetry_history = self.telemetry_history[-1000:]
        
        return telemetry
    
    def _check_errors(self, state, real_altitude):
        """
        Проверка ошибок в телеметрии.
        """
        errors = []
        sensors = state.get('sensors', {})
        
        # Проверка высоты
        if real_altitude > 50:
            errors.append({
                'sensor': 'altitude',
                'message': f'Слишком большая высота: {real_altitude:.1f} м',
                'value': real_altitude,
                'threshold': 50
            })
        
        # Проверка скорости
        vel = state.get('velocity', {})
        speed = math.sqrt(vel.get('x', 0)**2 + vel.get('y', 0)**2 + vel.get('z', 0)**2)
        if speed > 15:
            errors.append({
                'sensor': 'velocity',
                'message': f'Слишком большая скорость: {speed:.1f} м/с',
                'value': speed,
                'threshold': 15
            })
        
        # Проверка препятствий
        obstacle_dist = sensors.get('obstacle_distance', 100.0)
        if obstacle_dist < 5:
            errors.append({
                'sensor': 'obstacle',
                'message': f'Критическое приближение к препятствию: {obstacle_dist:.1f} м',
                'value': obstacle_dist,
                'threshold': 5
            })
        
        # Проверка батареи
        battery = sensors.get('battery', 100.0)
        if battery < 20:
            errors.append({
                'sensor': 'battery',
                'message': f'Низкий заряд батареи: {battery:.1f}%',
                'value': battery,
                'threshold': 20
            })
        
        return errors
    
    def get_position(self):
        """Возвращает последнюю позицию."""
        if self.last_telemetry:
            return self.last_telemetry.get('position', {})
        return {}
    
    def get_altitude(self):
        """Возвращает последнюю высоту."""
        pos = self.get_position()
        return pos.get('altitude', 0)
    
    def get_speed(self):
        """Возвращает последнюю скорость."""
        if self.last_telemetry:
            return self.last_telemetry.get('velocity', {}).get('speed', 0)
        return 0
    
    def get_obstacle_distance(self):
        """Возвращает расстояние до ближайшего препятствия."""
        if self.last_telemetry:
            return self.last_telemetry.get('sensors', {}).get('obstacle_distance', 100.0)
        return 100.0
    
    def get_sensor_summary(self):
        """Краткая сводка по сенсорам."""
        if not self.telemetry_history:
            return "Нет данных телеметрии"
        
        last = self.telemetry_history[-1]
        pos = last.get('position', {})
        vel = last.get('velocity', {})
        sensors = last.get('sensors', {})
        
        return {
            'position': f"({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f}, {pos.get('z', 0):.1f})",
            'altitude': f"{pos.get('altitude', 0):.1f} м",
            'speed': f"{vel.get('speed', 0):.1f} м/с",
            'obstacle': f"{sensors.get('obstacle_distance', 100.0):.1f} м",
            'battery': f"{sensors.get('battery', 100.0):.1f}%",
            'collision': "ДА" if sensors.get('collision', False) else "НЕТ",
            'errors_count': len(last.get('errors', []))
        }
    
    def get_telemetry_history(self, limit=100):
        """Возвращает историю телеметрии."""
        if limit <= 0:
            return self.telemetry_history
        return self.telemetry_history[-limit:]
    
    def clear_history(self):
        self.telemetry_history = []
        self.last_telemetry = None
        self.sensor_errors = []