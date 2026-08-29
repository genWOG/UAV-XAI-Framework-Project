"""
Модуль детекции препятствий для БПЛА.
Использует данные с сенсоров для определения препятствий.
"""

import math


class ObstacleDetector:
    """
    Детектор препятствий на основе данных с сенсоров.
    Определяет тип препятствия, расстояние и имя объекта.
    """

    def __init__(self, log_manager=None):
        self.log = log_manager
        self.last_detection = None
        self.detection_history = []

    def detect(self, state):
        """
        Детекция препятствий на основе состояния БПЛА.

        Parameters:
            state (dict): Состояние БПЛА от AirSim
                {
                    'position': {...},
                    'velocity': {...},
                    'orientation': {...},
                    'sensors': {
                        'obstacle_distance': float,
                        'obstacle_density': float,
                        'battery': float,
                        'collision': bool,
                        'object_name': str
                    }
                }

        Returns:
            dict or None: Информация о препятствии или None
                {
                    'type': 'collision' | 'large' | 'small' | 'environmental',
                    'object': str,
                    'distance': float,
                    'density': float
                }
        """
        sensors = state.get('sensors', {})
        
        # 👉 1. ПРОВЕРКА СТОЛКНОВЕНИЯ
        if sensors.get('collision', False):
            object_name = self._get_object_name(sensors)
            distance = sensors.get('obstacle_distance', 0.0)
            
            result = {
                'type': 'collision',
                'object': object_name,
                'distance': distance,
                'density': 0.0
            }
            self._log(f"🚨 СТОЛКНОВЕНИЕ с {object_name}!", level='error')
            self._save_detection(result)
            return result

        # 👉 2. ПОЛУЧАЕМ ДАННЫЕ С СЕНСОРОВ
        distance = sensors.get('obstacle_distance', 100.0)
        density = sensors.get('obstacle_density', 0.0)

        # Если расстояние очень большое — препятствий нет
        if distance > 50.0 and density < 0.1:
            self.last_detection = None
            return None

        # 👉 3. ОПРЕДЕЛЯЕМ ТИП ПРЕПЯТСТВИЯ
        result = {
            'distance': distance,
            'density': density,
            'object': self._get_object_name(sensors)
        }

        # Очень близкое препятствие (критическое)
        if distance < 3.0:
            result['type'] = 'collision'
            self._log(f"⚠️ КРИТИЧЕСКОЕ ПРИБЛИЖЕНИЕ! {result['object']} на {distance:.1f} м!", level='error')
        
        # Опасное препятствие (близкое)
        elif distance < 8.0:
            result['type'] = 'large'
            self._log(f"⚠️ ОПАСНОЕ ПРЕПЯТСТВИЕ! {result['object']} на {distance:.1f} м!", level='warn')
        
        # Среднее препятствие
        elif distance < 20.0:
            result['type'] = 'small'
            self._log(f"ℹ️ ПРЕПЯТСТВИЕ {result['object']} на {distance:.1f} м", level='info')
        
        # Высокая плотность препятствий
        elif density > 0.7:
            result['type'] = 'large'
            self._log(f"⚠️ СКОПЛЕНИЕ ПРЕПЯТСТВИЙ! Плотность {density:.0%}", level='warn')
        
        # Дальнее препятствие
        else:
            result['type'] = 'small'
            self._log(f"ℹ️ {result['object']} в {distance:.1f} м (обнаружено)", level='debug')

        self._save_detection(result)
        self.last_detection = result
        return result

    def _get_object_name(self, sensors):
        """
        Определяет имя объекта на основе данных с сенсоров.
        """
        # Если есть имя объекта от AirSim
        object_name = sensors.get('object_name', None)
        if object_name:
            return self._translate_object_name(object_name)

        # Если есть данные о типе объекта
        object_type = sensors.get('object_type', '')
        if object_type:
            return self._translate_object_name(object_type)

        # Определяем по расстоянию
        distance = sensors.get('obstacle_distance', 100.0)
        if distance < 3.0:
            return 'стена/здание'
        elif distance < 8.0:
            return 'препятствие'
        else:
            return 'объект'

    def _translate_object_name(self, name):
        """
        Переводит имя объекта с английского на русский.
        """
        if not name:
            return 'неизвестно'

        name_lower = name.lower()

        # Карта соответствий
        translation_map = {
            'building': 'здание',
            'house': 'здание',
            'wall': 'стена',
            'vehicle': 'транспорт',
            'car': 'машина',
            'bus': 'автобус',
            'truck': 'грузовик',
            'tree': 'дерево',
            'pole': 'столб',
            'lamp': 'фонарь',
            'sign': 'знак',
            'fence': 'забор',
            'barrier': 'барьер',
            'human': 'человек',
            'person': 'человек',
            'pedestrian': 'пешеход',
            'bicycle': 'велосипед',
            'motorcycle': 'мотоцикл',
            'drone': 'дрон',
            'bird': 'птица',
            'cloud': 'облако',
            'balloon': 'шар',
            'unknown': 'неизвестно'
        }

        for eng, rus in translation_map.items():
            if eng in name_lower:
                return rus

        return name

    def _log(self, message, level='info'):
        """Логирование"""
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
        """
        Сохраняет детекцию в историю.
        """
        self.detection_history.append({
            'timestamp': time.time() if 'time' in dir() else 0,
            'detection': detection
        })
        # Ограничиваем историю
        if len(self.detection_history) > 1000:
            self.detection_history = self.detection_history[-1000:]

    def get_summary(self):
        """
        Возвращает сводку по детекциям.
        """
        if not self.detection_history:
            return "Препятствий не обнаружено"

        total = len(self.detection_history)
        collision_count = sum(1 for d in self.detection_history 
                             if d.get('detection', {}).get('type') == 'collision')
        large_count = sum(1 for d in self.detection_history 
                         if d.get('detection', {}).get('type') == 'large')
        small_count = sum(1 for d in self.detection_history 
                         if d.get('detection', {}).get('type') == 'small')

        # Собираем статистику по объектам
        object_counts = {}
        for d in self.detection_history:
            obj = d.get('detection', {}).get('object', 'неизвестно')
            object_counts[obj] = object_counts.get(obj, 0) + 1

        # Топ-3 объекта
        top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_objects_str = ', '.join([f"{obj}: {count}" for obj, count in top_objects])

        return {
            'total_detections': total,
            'collision_count': collision_count,
            'large_count': large_count,
            'small_count': small_count,
            'top_objects': top_objects,
            'summary': f"Всего: {total}, Столкновений: {collision_count}, "
                      f"Опасных: {large_count}, Средних: {small_count}, "
                      f"Топ-3: {top_objects_str}"
        }

    def clear_history(self):
        """Очищает историю детекций"""
        self.detection_history = []
        self.last_detection = None


# Добавляем import time для _save_detection
import time