"""
Сканер мира для AirSim / Cosys-AirSim.
Собирает информацию об объектах в сцене и классифицирует их.
"""

import json
import os
import re

class WorldScanner:
    """Сканер и классификатор объектов в мире."""

    def __init__(self, airsim_client, log_manager=None):
        self.client = airsim_client
        self.log = log_manager
        self.objects = []          # Все объекты из мира
        self.obstacles = []        # Только препятствия
        self.ignore_list = []      # Что игнорировать
        self.object_map = {}       # {имя: тип}

    def scan_world(self):
        """Запрашивает список всех объектов в мире."""
        try:
            # Пробуем получить все объекты через симуляцию
            # В AirSim нет прямого метода, но есть simGetObjectPose
            # Мы можем получить список известных объектов
            self._log("🔍 Сканирование мира...", level='system')

            # Пробуем получить список через simListSceneObjects (если есть)
            try:
                objects = self.client.simListSceneObjects()
                if objects:
                    self.objects = objects
                    self._log(f"✅ Найдено {len(objects)} объектов в сцене")
                else:
                    self._fallback_scan()
            except:
                self._fallback_scan()

            # Классифицируем объекты
            self._classify_objects()
            self._log(f"📊 Из них препятствий: {len(self.obstacles)}")

        except Exception as e:
            self._log(f"⚠️ Ошибка сканирования мира: {e}", level='warn')
            self._create_default_world()

    def _fallback_scan(self):
        """Fallback: используем simGetObjectPose для известных объектов."""
        # Список общих префиксов объектов в AirSim
        common_prefixes = [
            'Building', 'House', 'Apartment', 'Tower', 'Cube',
            'Sphere', 'Cylinder', 'Tree', 'Foliage', 'Rock',
            'Vehicle', 'Car', 'Bus', 'Truck', 'TrafficLight',
            'Pole', 'Lamp', 'Sign', 'Fence', 'Wall', 'Barrier'
        ]

        # Пробуем получить позиции для каждого префикса
        found_objects = []
        for prefix in common_prefixes:
            for i in range(1, 100):
                obj_name = f"{prefix}_{i}"
                try:
                    pose = self.client.simGetObjectPose(obj_name)
                    if pose:
                        found_objects.append(obj_name)
                        self._log(f"  Найден: {obj_name}", level='debug')
                except:
                    pass

        self.objects = found_objects
        self._log(f"✅ Найдено {len(found_objects)} объектов (fallback)")

    def _classify_objects(self):
        """Классифицирует объекты по типам."""
        self.obstacles = []
        self.object_map = {}

        for obj in self.objects:
            obj_lower = obj.lower()
            obj_type = self._detect_type(obj_lower)

            # Сохраняем в карту
            self.object_map[obj] = obj_type

            # Если это препятствие --- добавляем
            if obj_type != 'ignore':
                self.obstacles.append({
                    'name': obj,
                    'type': obj_type,
                    'is_obstacle': True
                })

    def _detect_type(self, name):
        """Определяет тип объекта по имени."""
        # 👉 Список "опасных" типов (препятствия)
        obstacle_keywords = {
            'building': 'здание',
            'house': 'здание',
            'apartment': 'здание',
            'tower': 'здание',
            'wall': 'стена',
            'fence': 'забор',
            'barrier': 'барьер',
            'cube': 'объект',
            'sphere': 'объект',
            'cylinder': 'объект',
            'tree': 'дерево',
            'foliage': 'дерево',
            'rock': 'камень',
            'vehicle': 'транспорт',
            'car': 'машина',
            'bus': 'автобус',
            'truck': 'грузовик',
            'trafficlight': 'светофор',
            'traffic_light': 'светофор',
            'pole': 'столб',
            'lamp': 'фонарь',
            'sign': 'знак',
            'person': 'человек',
            'human': 'человек',
            'pedestrian': 'пешеход',
            'drone': 'дрон',
            'bird': 'птица'
        }

        # 👉 Список "безопасных" (игнорировать)
        ignore_keywords = [
            'landscape', 'ground', 'floor', 'terrain', 'sky',
            'cloud', 'sun', 'moon', 'star', 'background',
            'ceiling', 'door', 'window', 'glass', 'water',
            'road', 'street', 'sidewalk', 'path', 'grass'
        ]

        # Проверяем на игнорирование
        for keyword in ignore_keywords:
            if keyword in name:
                return 'ignore'

        # Проверяем на препятствие
        for keyword, obj_type in obstacle_keywords.items():
            if keyword in name:
                return obj_type

        # Если ничего не подошло --- считаем неизвестным препятствием
        return 'неизвестно'

    def get_obstacles(self):
        """Возвращает список препятствий."""
        return self.obstacles

    def get_object_type(self, name):
        """Возвращает тип объекта по имени."""
        return self.object_map.get(name, 'неизвестно')

    def is_obstacle(self, name):
        """Проверяет, является ли объект препятствием."""
        if name in self.object_map:
            return self.object_map[name] != 'ignore'
        return True  # По умолчанию считаем препятствием

    def update_from_log(self, object_name):
        """Обновляет карту из лога коллизий."""
        if object_name and object_name not in self.object_map:
            obj_type = self._detect_type(object_name.lower())
            self.object_map[object_name] = obj_type
            if obj_type != 'ignore':
                self.obstacles.append({
                    'name': object_name,
                    'type': obj_type,
                    'is_obstacle': True
                })
                self._log(f"📝 Добавлен объект: {object_name} ({obj_type})", level='debug')

    def get_obstacle_types(self):
        """Возвращает словарь с уникальными типами препятствий."""
        types = {}
        for obj in self.obstacles:
            t = obj['type']
            types[t] = types.get(t, 0) + 1
        return types

    def should_ignore(self, object_name):
        """Проверяет, нужно ли игнорировать объект."""
        if not object_name:
            return True
        obj_lower = object_name.lower()
        ignore_patterns = [
            'landscape', 'ground', 'floor', 'terrain',
            'sky', 'cloud', 'background'
        ]
        for pattern in ignore_patterns:
            if pattern in obj_lower:
                return True
        return False

    def _log(self, message, level='info'):
        if self.log:
            if level == 'system':
                self.log.system(message)
            elif level == 'warn':
                self.log.warn(message)
            elif level == 'error':
                self.log.error(message)
            else:
                self.log.info(message)
        else:
            print(f"[{level.upper()}] {message}")

    def _create_default_world(self):
        """Создаёт стандартную карту мира для Fallback."""
        self._log("⚠️ Использую стандартную карту мира", level='warn')
        default_objects = [
            'Building', 'House', 'Apartment', 'Tower',
            'Tree', 'Foliage', 'Rock', 'Vehicle',
            'Car', 'Bus', 'TrafficLight', 'Pole'
        ]
        self.objects = default_objects
        self._classify_objects()

    def save_to_file(self, filepath="data/world_objects.json"):
        """Сохраняет карту объектов в файл."""
        try:
            data = {
                'objects': self.objects,
                'obstacles': self.obstacles,
                'object_map': self.object_map,
                'types': self.get_obstacle_types()
            }
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._log(f"✅ Карта мира сохранена: {filepath}", level='system')
        except Exception as e:
            self._log(f"❌ Ошибка сохранения карты: {e}", level='error')

    def load_from_file(self, filepath="data/world_objects.json"):
        """Загружает карту объектов из файла."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.objects = data.get('objects', [])
            self.obstacles = data.get('obstacles', [])
            self.object_map = data.get('object_map', {})
            self._log(f"✅ Карта мира загружена: {filepath}", level='system')
            return True
        except Exception as e:
            self._log(f"⚠️ Не удалось загрузить карту: {e}", level='warn')
            return False