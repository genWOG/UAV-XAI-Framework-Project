"""
Модуль детекции препятствий для БПЛА.
Использует данные с сенсоров и позицию для определения препятствий.
"""

import time
import math

class ObstacleDetector:
    """
    Детектор препятствий на основе данных с сенсоров.
    """

    def __init__(self, log_manager=None):
        self.log = log_manager
        self.last_detection = None
        self.detection_history = []
        self.obstacle_positions = {}
        self.last_altitude = 0
        self.altitude_change_counter = 0

    def detect(self, state):
        """
        Детекция препятствий на основе состояния БПЛА.
        """
        sensors = state.get('sensors', {})
        pos = state.get('position', {})
        
        # 👉 РЕАЛЬНАЯ ВЫСОТА
        real_altitude = abs(pos.get('z', 0))
        
        # 👉 1. ПРОВЕРКА СТОЛКНОВЕНИЯ
        if sensors.get('collision', False):
            object_name = sensors.get('object_name', 'неизвестно')
            distance = sensors.get('obstacle_distance', 0.0)
            object_name = self._translate_object_name(object_name)
            
            if 'дерево' in object_name.lower() or 'tree' in object_name.lower():
                return None
            
            result = {
                'type': 'collision',
                'object': object_name,
                'distance': distance,
                'density': 0.0,
                'position': pos,
                'altitude': real_altitude
            }
            self._log(f"🚨 СТОЛКНОВЕНИЕ с {object_name}!", level='error')
            self._save_detection(result)
            return result

        # 👉 2. ПОЛУЧАЕМ ДАННЫЕ С СЕНСОРОВ
        distance = sensors.get('obstacle_distance', 100.0)
        density = sensors.get('obstacle_density', 0.0)
        object_name = sensors.get('object_name', None)
        
        if object_name:
            object_name = self._translate_object_name(object_name)
        else:
            object_name = 'неизвестно'

        # 👉 ИГНОРИРУЕМ ДЕРЕВЬЯ
        if 'дерево' in object_name.lower() or 'tree' in object_name.lower():
            return None

        # 👉 3. АНАЛИЗ ИЗМЕНЕНИЯ ВЫСОТЫ (признак обхода)
        altitude_change = real_altitude - self.last_altitude
        self.last_altitude = real_altitude
        
        # Если высота резко меняется — возможно, обходит препятствие
        if abs(altitude_change) > 5:
            self.altitude_change_counter += 1
            if self.altitude_change_counter > 3:
                self._log(f"📈 Обнаружено изменение высоты: {altitude_change:.1f} м", level='debug')
                # Это не препятствие, а обход
                return None
        else:
            self.altitude_change_counter = 0

        # 👉 4. ПРОВЕРКА ПО РАССТОЯНИЮ (используем DistanceSensor)
        if distance < 50.0:
            # Определяем тип по расстоянию
            if distance < 5.0:
                result_type = 'collision'
                log_level = 'error'
                log_msg = f"⚠️ КРИТИЧЕСКОЕ ПРИБЛИЖЕНИЕ! {object_name} на {distance:.1f} м!"
            elif distance < 15.0:
                result_type = 'large'
                log_level = 'warn'
                log_msg = f"⚠️ ОПАСНОЕ ПРЕПЯТСТВИЕ! {object_name} на {distance:.1f} м!"
            elif distance < 30.0:
                result_type = 'medium'
                log_level = 'info'
                log_msg = f"ℹ️ ПРЕПЯТСТВИЕ {object_name} на {distance:.1f} м"
            else:
                result_type = 'small'
                log_level = 'info'
                log_msg = f"ℹ️ {object_name} на расстоянии {distance:.1f} м"
            
            # Если объект неизвестен, пытаемся угадать
            if object_name == 'неизвестно' or object_name == 'Unknown':
                guess = self._guess_object_type(distance, density, real_altitude)
                object_name = guess
                log_msg = f"🔍 ПРЕДПОЛОЖЕНИЕ: {guess} ({distance:.1f} м)"
            
            result = {
                'type': result_type,
                'object': object_name,
                'distance': distance,
                'density': density,
                'position': pos,
                'altitude': real_altitude
            }
            
            self._log(log_msg, level=log_level)
            self._save_detection(result)
            self.last_detection = result
            return result

        # 👉 5. ПРОВЕРКА ПО ПЛОТНОСТИ
        if density > 0.5:
            result = {
                'type': 'large',
                'object': 'скопление объектов',
                'distance': distance,
                'density': density,
                'position': pos,
                'altitude': real_altitude
            }
            self._log(f"⚠️ СКОПЛЕНИЕ ПРЕПЯТСТВИЙ! Плотность {density:.0%}", level='warn')
            self._save_detection(result)
            self.last_detection = result
            return result

        # 👉 6. ПРОВЕРКА ПО ИСТОРИИ
        pos_key = (round(pos.get('x', 0), 1), round(pos.get('y', 0), 1))
        if pos_key in self.obstacle_positions:
            old_obj = self.obstacle_positions[pos_key]
            result = {
                'type': 'small',
                'object': old_obj,
                'distance': 30.0,
                'density': 0.3,
                'position': pos,
                'altitude': real_altitude
            }
            self._log(f"ℹ️ Вспоминаем {old_obj} в этой точке", level='debug')
            self._save_detection(result)
            self.last_detection = result
            return result

        self.last_detection = None
        return None

    def _guess_object_type(self, distance, density, altitude):
        """Угадывает тип объекта по характеристикам."""
        if distance < 5.0:
            if altitude > 10:
                return 'здание/сооружение'
            return 'здание/стена'
        elif distance < 15.0 and density > 0.3:
            return 'крупный объект'
        elif distance < 15.0:
            return 'препятствие'
        elif distance < 30.0 and density > 0.5:
            return 'скопление объектов'
        elif distance < 30.0:
            return 'объект'
        else:
            return 'неизвестно'

    def _translate_object_name(self, name):
        """Переводит имя объекта с английского на русский."""
        if not name:
            return 'неизвестно'
        
        name_lower = name.lower()
        translation_map = {
            'building': 'здание',
            'house': 'здание',
            'apartment': 'здание',
            'tower': 'здание',
            'square': 'здание',
            'wall': 'стена',
            'fence': 'забор',
            'barrier': 'барьер',
            'vehicle': 'транспорт',
            'car': 'машина',
            'bus': 'автобус',
            'truck': 'грузовик',
            'motorcycle': 'мотоцикл',
            'bicycle': 'велосипед',
            'npcvehicle': 'транспорт',
            'vehiclemanager': 'автобус',
            'tree': 'дерево',
            'foliage': 'дерево',
            'pole': 'столб',
            'lamp': 'фонарь',
            'sign': 'знак',
            'human': 'человек',
            'person': 'человек',
            'pedestrian': 'пешеход',
            'drone': 'дрон',
            'bird': 'птица',
            'cloud': 'облако',
            'balloon': 'шар',
            'trafficlight': 'светофор',
            'traffic_light': 'светофор',
            'wire': 'провод',
            'cable': 'провод',
            'powerline': 'провод',
            'electric': 'провод',
            'apartment_square_tower2_107': 'здание',
            'npcvehiclemanager_20': 'автобус',
            'CityEnviron_TrafficLight59_3058': 'светофор',
            'unknown': 'неизвестно'
        }
        
        for eng, rus in translation_map.items():
            if eng in name_lower:
                return rus
        return name

    def _log(self, message, level='info'):
        if self.log:
            if level == 'error':
                self.log.error(f"🛑 {message}")
            elif level == 'warn':
                self.log.warn(f"⚠️ {message}")
            elif level == 'debug':
                self.log.info(f"🔍 {message}")
            else:
                self.log.info(f"ℹ️ {message}")
        else:
            print(f"[{level.upper()}] {message}")

    def _save_detection(self, detection):
        self.detection_history.append({
            'timestamp': time.time(),
            'detection': detection
        })
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]
        
        # Запоминаем позицию
        pos = detection.get('position', {})
        if pos:
            pos_key = (round(pos.get('x', 0), 1), round(pos.get('y', 0), 1))
            self.obstacle_positions[pos_key] = detection.get('object', 'препятствие')

    def get_summary(self):
        if not self.detection_history:
            return "Препятствий не обнаружено"
        
        total = len(self.detection_history)
        collision_count = sum(1 for d in self.detection_history 
                            if d.get('detection', {}).get('type') == 'collision')
        large_count = sum(1 for d in self.detection_history 
                        if d.get('detection', {}).get('type') == 'large')
        medium_count = sum(1 for d in self.detection_history 
                          if d.get('detection', {}).get('type') == 'medium')
        small_count = sum(1 for d in self.detection_history 
                         if d.get('detection', {}).get('type') == 'small')
        
        object_counts = {}
        for d in self.detection_history:
            obj = d.get('detection', {}).get('object', 'неизвестно')
            object_counts[obj] = object_counts.get(obj, 0) + 1
        
        top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_objects_str = ', '.join([f"{obj}: {count}" for obj, count in top_objects])
        
        return {
            'total_detections': total,
            'collision_count': collision_count,
            'large_count': large_count,
            'medium_count': medium_count,
            'small_count': small_count,
            'top_objects': top_objects,
            'summary': f"Всего: {total}, Столкновений: {collision_count}, "
                      f"Опасных: {large_count}, Средних: {medium_count}, "
                      f"Маленьких: {small_count}, Топ-3: {top_objects_str}"
        }

    def clear_history(self):
        self.detection_history = []
        self.last_detection = None
        self.obstacle_positions = {}
        self.altitude_change_counter = 0