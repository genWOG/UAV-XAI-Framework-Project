"""
Модуль детекции препятствий для БПЛА.

Использует встроенную систему коллизий AirSim + карту мира.
"""

import time

class ObstacleDetector:
    """Детектор препятствий на основе состояния БПЛА."""

    def __init__(self, log_manager=None, world_scanner=None):
        self.log = log_manager
        self.world_scanner = world_scanner
        self.last_detection = None
        self.detection_history = []
        self.obstacle_positions = {}
        
        # 👉 СПИСОК ОБЪЕКТОВ, КОТОРЫЕ ВСЕГДА ИГНОРИРУЕМ
        self.always_ignore = [
            'landscape', 'ground', 'floor', 'terrain',
            'sky', 'cloud', 'background', 'grass',
            'road', 'street', 'sidewalk', 'path'
        ]

    def detect(self, state):
        """
        Детекция препятствий на основе состояния БПЛА.
        Использует collision_info из AirSim.
        """
        sensors = state.get('sensors', {})
        pos = state.get('position', {})

        # 👉 1. ПРОВЕРКА КОЛЛИЗИИ (ГЛАВНЫЙ ИСТОЧНИК!)
        if sensors.get('collision', False):
            object_name = sensors.get('object_name', 'неизвестно')
            distance = sensors.get('obstacle_distance', 0.0)

            # 👉 ПРОВЕРЯЕМ, НУЖНО ЛИ ИГНОРИРОВАТЬ
            if self._should_ignore(object_name):
                self._log(f"ℹ️ Игнорируем: {object_name}", level='debug')
                return None

            # Переводим имя на русский
            object_name = self._translate_object_name(object_name)

            # Игнорируем деревья (если они не коллизионные)
            if 'дерево' in object_name.lower() or 'tree' in object_name.lower():
                return None

            result = {
                'type': 'collision',
                'object': object_name,
                'distance': distance,
                'density': 0.0,
                'position': pos,
                'is_obstacle': True
            }

            self._log(f"🚨 СТОЛКНОВЕНИЕ с {object_name}!", level='error')
            self._save_detection(result)
            return result

        # 👉 2. ЕСЛИ НЕТ КОЛЛИЗИИ, ПРОВЕРЯЕМ DISTANCE SENSOR (FALLBACK)
        distance = sensors.get('obstacle_distance', 100.0)
        density = sensors.get('obstacle_density', 0.0)
        object_name = sensors.get('object_name', None)

        if object_name:
            # 👉 ПРОВЕРКА НА ИГНОРИРОВАНИЕ
            if self._should_ignore(object_name):
                return None
            object_name = self._translate_object_name(object_name)
        else:
            object_name = 'неизвестно'

        # Игнорируем деревья
        if 'дерево' in object_name.lower() or 'tree' in object_name.lower():
            return None

        # Если расстояние меньше 30 метров --- это препятствие
        if distance < 30.0:
            if distance < 5.0:
                result_type = 'collision'
                log_level = 'error'
                log_msg = f"⚠️ КРИТИЧЕСКОЕ ПРИБЛИЖЕНИЕ! {object_name} на {distance:.1f} м!"
            elif distance < 15.0:
                result_type = 'large'
                log_level = 'warn'
                log_msg = f"⚠️ ОПАСНОЕ ПРЕПЯТСТВИЕ! {object_name} на {distance:.1f} м!"
            else:
                result_type = 'small'
                log_level = 'info'
                log_msg = f"ℹ️ {object_name} на расстоянии {distance:.1f} м"

            # Если объект неизвестен, пытаемся угадать
            if object_name == 'неизвестно' or object_name == 'Unknown':
                guess = self._guess_object_type(distance, density)
                object_name = guess
                log_msg = f"🔍 ПРЕДПОЛОЖЕНИЕ: {guess} ({distance:.1f} м)"

            result = {
                'type': result_type,
                'object': object_name,
                'distance': distance,
                'density': density,
                'position': pos,
                'is_obstacle': True
            }

            self._log(log_msg, level=log_level)
            self._save_detection(result)
            self.last_detection = result
            return result

        # 👉 3. ПРОВЕРКА ПО ИСТОРИИ (если уже видели препятствие в этом месте)
        pos_key = (round(pos.get('x', 0), 1), round(pos.get('y', 0), 1))
        if pos_key in self.obstacle_positions:
            old_obj = self.obstacle_positions[pos_key]
            # Проверяем, не игнорируется ли этот объект
            if self._should_ignore(old_obj):
                return None
            result = {
                'type': 'small',
                'object': old_obj,
                'distance': 30.0,
                'density': 0.3,
                'position': pos,
                'is_obstacle': True
            }
            self._log(f"ℹ️ Вспоминаем {old_obj} в этой точке", level='debug')
            self._save_detection(result)
            self.last_detection = result
            return result

        self.last_detection = None
        return None

    def _should_ignore(self, object_name):
        """Проверяет, нужно ли игнорировать объект."""
        if not object_name:
            return True
        
        object_lower = object_name.lower()
        
        # Проверяем по списку всегда игнорируемых
        for pattern in self.always_ignore:
            if pattern in object_lower:
                return True
        
        # Если есть сканер мира, используем его
        if self.world_scanner:
            return self.world_scanner.should_ignore(object_name)
        
        return False

    def _guess_object_type(self, distance, density):
        """Угадывает тип объекта по характеристикам."""
        if distance < 5.0:
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
            'unknown': 'неизвестно'
        }

        # 👉 ОБРАБОТКА СПЕЦИАЛЬНЫХ ШАБЛОНОВ
        if 'templatefloor' in name_lower:
            return 'здание'
        if 'cube' in name_lower and 'landscape' not in name_lower:
            return 'препятствие'
        if 'landscape' in name_lower:
            return 'ландшафт'  # будет проигнорирован

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

    def set_world_scanner(self, scanner):
        """Устанавливает сканер мира для фильтрации объектов."""
        self.world_scanner = scanner

    def get_summary(self):
        if not self.detection_history:
            return "Препятствий не обнаружено"

        total = len(self.detection_history)
        collision_count = sum(1 for d in self.detection_history
                              if d.get('detection', {}).get('type') == 'collision')
        large_count = sum(1 for d in self.detection_history
                          if d.get('detection', {}).get('type') == 'large')
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
            'small_count': small_count,
            'top_objects': top_objects,
            'summary': f"Всего: {total}, Столкновений: {collision_count}, "
                       f"Опасных: {large_count}, Маленьких: {small_count}, Топ-3: {top_objects_str}"
        }

    def clear_history(self):
        self.detection_history = []
        self.last_detection = None
        self.obstacle_positions = {}